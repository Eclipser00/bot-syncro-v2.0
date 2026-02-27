"""Servicio de visualizacion automatica integrado al loop del bot."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bot_trading.domain.entities import SymbolConfig

from .event_reader import EventReader
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

    def __post_init__(self) -> None:
        self.refresh_seconds = max(1, int(self.refresh_seconds))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_store = VisualizerStateStore(symbols=[s.name for s in self.symbols])
        self.event_reader = EventReader(
            pivot_log_path=self.pivot_log_path,
            bot_events_path=self.bot_events_path,
            start_from_end=self.start_from_end,
        )
        self.next_refresh = datetime.now(timezone.utc)
        logger.info(
            "AutoVisualizer activo (refresh=%ss, output_dir=%s, events=%s)",
            self.refresh_seconds,
            self.output_dir,
            self.bot_events_path,
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
            self.render_all()
            self.next_refresh = current_time + timedelta(seconds=self.refresh_seconds)

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

