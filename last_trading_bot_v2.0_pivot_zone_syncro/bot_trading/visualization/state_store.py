"""Estado incremental en memoria del visualizador M3."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_ENTRY_EVENT_TYPES = {"signal", "order_fill", "position"}


def _to_utc_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _project_to_m3(ts: pd.Timestamp) -> pd.Timestamp:
    """Proyecta timestamp al eje temporal M3 (etiqueta de cierre)."""
    return ts.ceil("3min")


@dataclass
class SymbolState:
    candles_m3: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    )
    saved_zones: list[dict[str, Any]] = field(default_factory=list)
    active_zone: dict[str, Any] | None = None
    entry_events: list[dict[str, Any]] = field(default_factory=list)
    seen_zone_keys: set[tuple[Any, ...]] = field(default_factory=set)


class VisualizerStateStore:
    """Buffer incremental por símbolo para velas, zonas y eventos de entrada."""

    def __init__(
        self,
        symbols: list[str],
        max_candles: int = 3000,
        max_events: int = 1500,
        max_zones: int = 300,
    ) -> None:
        self.max_candles = max_candles
        self.max_events = max_events
        self.max_zones = max_zones
        self._states: dict[str, SymbolState] = {symbol: SymbolState() for symbol in symbols}

    def _ensure_symbol(self, symbol: str) -> SymbolState:
        if symbol not in self._states:
            self._states[symbol] = SymbolState()
        return self._states[symbol]

    def update_candles(self, symbol: str, candles_m3: pd.DataFrame) -> None:
        if candles_m3 is None or candles_m3.empty:
            return

        state = self._ensure_symbol(symbol)
        frame = candles_m3.copy()
        idx = pd.DatetimeIndex(frame.index)
        if idx.tz is None:
            frame.index = idx.tz_localize("UTC")
        else:
            frame.index = idx.tz_convert("UTC")
        frame = frame.sort_index()

        if state.candles_m3.empty:
            merged = frame
        else:
            merged = pd.concat([state.candles_m3, frame]).sort_index()
            merged = merged.loc[~merged.index.duplicated(keep="last")]
        state.candles_m3 = merged.tail(self.max_candles)

    def update_candles_batch(self, candles_by_symbol: dict[str, pd.DataFrame]) -> None:
        for symbol, df in candles_by_symbol.items():
            self.update_candles(symbol, df)

    def ingest_pivot_zones(self, pivot_updates: list[dict[str, Any]]) -> None:
        for zone in pivot_updates:
            symbol = str(zone.get("symbol", "UNKNOWN"))
            state = self._ensure_symbol(symbol)

            ts = _to_utc_timestamp(zone.get("ts_event"))
            if ts is None:
                continue

            key = (
                ts.isoformat(),
                round(float(zone.get("top", 0.0)), 8),
                round(float(zone.get("bot", 0.0)), 8),
                int(zone.get("bar", -1)),
            )
            if key in state.seen_zone_keys:
                continue

            state.seen_zone_keys.add(key)
            state.saved_zones.append(
                {
                    "ts_event": ts,
                    "top": float(zone.get("top", 0.0)),
                    "bot": float(zone.get("bot", 0.0)),
                    "mid": float(zone.get("mid", 0.0)),
                    "width": float(zone.get("width", 0.0)),
                    "bar": int(zone.get("bar", -1)),
                    "pivots": int(zone.get("pivots", 0)),
                    "total_saved": int(zone.get("total_saved", 0)),
                }
            )
            if len(state.saved_zones) > self.max_zones:
                state.saved_zones = state.saved_zones[-self.max_zones :]

    def ingest_bot_events(self, bot_updates: list[dict[str, Any]]) -> None:
        for event in bot_updates:
            symbol = str(event.get("symbol", "UNKNOWN"))
            state = self._ensure_symbol(symbol)

            event_type = str(event.get("event_type", "misc"))
            ts = _to_utc_timestamp(event.get("ts_event"))
            if ts is None:
                continue

            if event_type == "zone_state":
                state.active_zone = {
                    "ts_event": ts,
                    "mid": float(event.get("state_mid", 0.0)),
                    "width": float(event.get("state_width", 0.0)),
                    "valid": int(event.get("state_valid", 0)),
                }
                continue

            if event_type not in _ENTRY_EVENT_TYPES:
                continue

            projected = _project_to_m3(ts)
            entry = {
                "ts_event": ts,
                "ts_m3": projected,
                "event_type": event_type,
                "price": float(event.get("price", 0.0)),
                "size": float(event.get("size", 0.0)),
                "side": str(event.get("side", "flat")),
                "reason": str(event.get("reason", "")),
                "order_id": str(event.get("order_id", "")),
                "bar_index": int(event.get("bar_index", -1)),
                "timeframe": str(event.get("timeframe", "")),
            }
            state.entry_events.append(entry)
            if len(state.entry_events) > self.max_events:
                state.entry_events = state.entry_events[-self.max_events :]

    def get_states(self) -> dict[str, SymbolState]:
        return self._states

    def get_symbol_state(self, symbol: str) -> SymbolState | None:
        return self._states.get(symbol)

    def latest_timestamp(self) -> pd.Timestamp | None:
        latest: pd.Timestamp | None = None
        for state in self._states.values():
            if state.candles_m3.empty:
                continue
            ts = pd.Timestamp(state.candles_m3.index[-1])
            if ts.tzinfo is None:
                ts = ts.tz_localize(timezone.utc)
            if latest is None or ts > latest:
                latest = ts
        return latest
