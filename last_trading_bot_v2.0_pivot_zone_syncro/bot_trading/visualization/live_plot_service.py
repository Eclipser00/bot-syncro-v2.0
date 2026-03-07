"""Servicio de visualizacion automatica integrado al loop del bot."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from bot_trading.domain.entities import SymbolConfig, TradeRecord

from .event_reader import EventReader, parse_bot_event_line, parse_pivot_zone_line
from .plot_bokeh import BokehPlotBuilder
from .state_store import VisualizerStateStore

logger = logging.getLogger(__name__)


def resolve_bot_events_path(root: Path, configured_path: str) -> Path:
    """Resuelve ruta de eventos local o en carpeta padre, priorizando el mas reciente."""
    local_path = (root / configured_path).resolve()
    parent_path = (root.parent / configured_path).resolve()

    local_exists = local_path.exists()
    parent_exists = parent_path.exists()

    if local_exists and parent_exists:
        return parent_path if parent_path.stat().st_mtime > local_path.stat().st_mtime else local_path
    if parent_exists:
        return parent_path
    return local_path


@dataclass
class AutoVisualizerService:
    """Mantiene estado M3 por simbolo y renderiza HTMLs por ticker."""

    symbols: list[SymbolConfig]
    output_dir: Path
    pivot_log_path: Path
    bot_events_path: Path
    refresh_seconds: int = 5
    start_from_end: bool = True
    max_candles: int | None = None
    closed_trades_provider: Callable[[datetime, datetime], list[TradeRecord]] | None = None
    closed_trades_lookback_days: int = 15

    def __post_init__(self) -> None:
        self.refresh_seconds = max(1, int(self.refresh_seconds))
        self.closed_trades_lookback_days = max(1, int(self.closed_trades_lookback_days))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.max_candles is not None and int(self.max_candles) <= 0:
            self.max_candles = None
        self.state_store = VisualizerStateStore(symbols=[s.name for s in self.symbols], max_candles=self.max_candles)
        self.next_refresh = datetime.now(timezone.utc)
        self._has_rendered_once = False
        if self.start_from_end:
            self._bootstrap_completed = True
        else:
            self._bootstrap_completed = False
            self._bootstrap_history(self.next_refresh)
            self._bootstrap_completed = True
        self.event_reader = EventReader(
            pivot_log_path=self.pivot_log_path,
            bot_events_path=self.bot_events_path,
            start_from_end=True,
        )
        # Primer read para fijar offsets al final actual y no saltar eventos
        # que se escriban entre el arranque y el primer callback por simbolo.
        try:
            self.event_reader.read_updates()
        except Exception as exc:
            logger.debug("AutoVisualizer no pudo primar offsets iniciales: %s", exc)
        logger.info(
            "AutoVisualizer activo (refresh=%ss, output_dir=%s, events=%s, start_from_end=%s)",
            self.refresh_seconds,
            self.output_dir,
            self.bot_events_path,
            self.start_from_end,
        )

    def on_market_data(
        self,
        symbol: SymbolConfig,
        data_by_timeframe: dict,
        current_time: datetime,
    ) -> None:
        m3 = data_by_timeframe.get("M3")
        if m3 is not None and not m3.empty:
            self.state_store.update_candles(symbol.name, m3)

        pivot_updates, bot_updates = self.event_reader.read_updates()
        if pivot_updates:
            self.state_store.ingest_pivot_zones(pivot_updates)
        if bot_updates:
            self.state_store.ingest_bot_events(bot_updates)

        if current_time >= self.next_refresh:
            # Evita primer render parcial: en el arranque los callbacks llegan por símbolo
            # y sin este guard se puede sobrescribir un HTML con estado incompleto.
            if not self._has_rendered_once:
                states = self.state_store.get_states()
                missing_candles = [
                    sym.name
                    for sym in self.symbols
                    if sym.name not in states or states[sym.name].candles_m3.empty
                ]
                if missing_candles:
                    logger.debug(
                        "AutoVisualizer difiere primer render; faltan velas para: %s",
                        ",".join(missing_candles),
                    )
                    return

            self.render_all()
            self.next_refresh = current_time + timedelta(seconds=self.refresh_seconds)
            self._has_rendered_once = True

    def _bootstrap_history(self, current_time: datetime) -> None:
        """Bootstrap inicial cuando start_from_end=False."""
        pivot_updates = self._read_historical_pivots()
        if pivot_updates:
            self.state_store.ingest_pivot_zones(pivot_updates)
        logger.info("AutoVisualizer bootstrap pivots_loaded=%d", len(pivot_updates))

        signal_updates = self._read_historical_signals()
        if signal_updates:
            self.state_store.ingest_bot_events(signal_updates)
        logger.info(
            "AutoVisualizer bootstrap signals_loaded=%d",
            len(signal_updates),
        )

        snapshot_events = self._build_mt5_snapshot_events(current_time)
        if snapshot_events:
            self.state_store.ingest_bot_events(snapshot_events)

    def _read_historical_pivots(self) -> list[dict]:
        updates: list[dict] = []
        if not self.pivot_log_path.exists():
            return updates
        try:
            with self.pivot_log_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    parsed = parse_pivot_zone_line(line.strip())
                    if parsed is not None:
                        updates.append(parsed)
        except Exception as exc:
            logger.warning("AutoVisualizer bootstrap no pudo leer pivots historicos: %s", exc)
        return updates

    def _read_historical_signals(self) -> list[dict]:
        updates: list[dict] = []
        if not self.bot_events_path.exists():
            return updates
        try:
            with self.bot_events_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    parsed = parse_bot_event_line(line.strip())
                    if parsed is None:
                        continue
                    if str(parsed.get("event_type")) != "signal":
                        continue
                    updates.append(parsed)
        except Exception as exc:
            logger.warning("AutoVisualizer bootstrap no pudo leer signals historicas: %s", exc)
        return updates

    def _build_mt5_snapshot_events(self, current_time: datetime) -> list[dict]:
        if self.closed_trades_provider is None:
            logger.info("AutoVisualizer bootstrap mt5_snapshot_disabled=1")
            return []

        to_utc = self._to_utc(current_time)
        from_utc = to_utc - timedelta(days=self.closed_trades_lookback_days)
        try:
            trades = list(self.closed_trades_provider(from_utc, to_utc) or [])
        except Exception as exc:
            logger.warning("AutoVisualizer bootstrap snapshot MT5 fallido: %s", exc)
            return []

        events: list[dict] = []
        for trade in trades:
            events.extend(self._trade_to_bootstrap_events(trade))

        logger.info(
            "AutoVisualizer bootstrap mt5_snapshot range_utc=%s..%s trades=%d events=%d",
            from_utc.isoformat(),
            to_utc.isoformat(),
            len(trades),
            len(events),
        )
        return events

    def _trade_to_bootstrap_events(self, trade: TradeRecord) -> list[dict]:
        symbol = str(getattr(trade, "symbol", "") or "").strip()
        if not symbol:
            return []

        entry_price = float(getattr(trade, "entry_price", 0.0) or 0.0)
        exit_price = float(getattr(trade, "exit_price", 0.0) or 0.0)
        size = abs(float(getattr(trade, "size", 0.0) or 0.0))
        if entry_price <= 0.0 or exit_price <= 0.0 or size <= 0.0:
            return []

        entry_time = self._to_utc(trade.entry_time)
        exit_time = self._to_utc(trade.exit_time)
        side = self._infer_trade_side(trade)
        position_ref = (
            str(getattr(trade, "position_id", "") or "")
            or str(getattr(trade, "exit_deal_ticket", "") or "")
            or str(getattr(trade, "entry_deal_ticket", "") or "")
        )

        order_open = {
            "event_type": "order_fill",
            "symbol": symbol,
            "ts_event": entry_time.isoformat(),
            "side": side,
            "price": entry_price,
            "size": size,
            "reason": "bootstrap_mt5_entry",
            "order_id": f"bootstrap:{position_ref}:open",
            "bar_index": -1,
            "timeframe": "M3",
        }
        order_close = {
            "event_type": "order_fill",
            "symbol": symbol,
            "ts_event": exit_time.isoformat(),
            "side": "flat",
            "price": exit_price,
            "size": size,
            "reason": "close_bootstrap_mt5_snapshot",
            "order_id": f"bootstrap:{position_ref}:close",
            "bar_index": -1,
            "timeframe": "M3",
        }
        position_flat = {
            "event_type": "position",
            "symbol": symbol,
            "ts_event": exit_time.isoformat(),
            "side": "flat",
            "price": exit_price,
            "size": size,
            "reason": "bootstrap_mt5_position_flat",
            "order_id": f"bootstrap:{position_ref}:position_flat",
            "bar_index": -1,
            "timeframe": "M3",
        }
        return [order_open, order_close, position_flat]

    @staticmethod
    def _infer_trade_side(trade: TradeRecord) -> str:
        """Infiera lado de apertura usando consistencia entre delta de precio y PnL."""
        entry_price = float(getattr(trade, "entry_price", 0.0) or 0.0)
        exit_price = float(getattr(trade, "exit_price", 0.0) or 0.0)
        pnl = float(getattr(trade, "pnl", 0.0) or 0.0)
        delta = exit_price - entry_price
        if abs(delta) < 1e-12:
            return "long" if pnl >= 0.0 else "short"
        return "long" if (delta * pnl) >= 0.0 else "short"

    @staticmethod
    def _to_utc(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    def render_all(self) -> None:
        states = self.state_store.get_states()
        for symbol, state in states.items():
            if state.candles_m3.empty:
                continue
            output_path = self.output_dir / f"visualizer{symbol}.html"
            builder = BokehPlotBuilder(
                output_path=output_path,
                title=f"Visualizer {symbol} - M3",
            )
            ok = builder.render({symbol: state})
            if ok:
                logger.info("Plot actualizado: %s", output_path)

    def close(self) -> None:
        try:
            self.render_all()
        except Exception as exc:
            logger.debug("No se pudo renderizar en cierre del AutoVisualizer: %s", exc)

