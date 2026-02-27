"""Lectura incremental de logs/eventos para el visualizador."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SYMBOL_RE = re.compile(r"(?:OANDA_)?([A-Z]{6})")


def normalize_symbol(raw: str | None) -> str:
    """Normaliza símbolos de eventos a un formato compacto (ej. EURUSD)."""
    if not raw:
        return "UNKNOWN"
    text = str(raw).strip()
    match = _SYMBOL_RE.search(text.upper())
    if match:
        return match.group(1)
    text = text.split(",")[0].strip()
    text = text.replace("OANDA_", "")
    return text or "UNKNOWN"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


@dataclass
class IncrementalFileReader:
    """Lee líneas nuevas de un archivo manteniendo offset interno."""

    path: Path
    offset: int = 0
    start_from_end: bool = False
    _initialized: bool = False

    def read_new_lines(self) -> list[str]:
        if not self.path.exists():
            self.offset = 0
            self._initialized = False
            return []

        size = self.path.stat().st_size
        if not self._initialized:
            self._initialized = True
            if self.start_from_end and self.offset == 0:
                self.offset = size
                return []

        if size < self.offset:
            # Archivo truncado/rotado: reiniciar lectura desde el inicio.
            self.offset = 0

        if size == self.offset:
            return []

        with self.path.open("r", encoding="utf-8") as fh:
            fh.seek(self.offset)
            lines = fh.readlines()
            self.offset = fh.tell()

        return [line.rstrip("\n") for line in lines if line.strip()]


def parse_pivot_zone_line(line: str) -> dict[str, Any] | None:
    """Parsea una línea de `pivot_zones.log` a un dict homogéneo."""
    tokens = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        tokens[key] = value

    if "time" not in tokens or "symbol" not in tokens:
        return None

    top = _to_float(tokens.get("top"))
    bot = _to_float(tokens.get("bot"))
    return {
        "event_type": "pivot_zone",
        "ts_event": tokens.get("time"),
        "symbol": normalize_symbol(tokens.get("symbol")),
        "timeframe": tokens.get("tf", "M3"),
        "top": top,
        "bot": bot,
        "mid": _to_float(tokens.get("mid"), (top + bot) / 2.0),
        "width": _to_float(tokens.get("width"), abs(top - bot) / 2.0),
        "bar": _to_int(tokens.get("bar"), -1),
        "pivots": _to_int(tokens.get("pivots"), 0),
        "total_saved": _to_int(tokens.get("total_saved"), 0),
    }


def parse_bot_event_line(line: str) -> dict[str, Any] | None:
    """Parsea una línea JSONL de `bot_events.jsonl`."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    payload["symbol"] = normalize_symbol(payload.get("symbol"))
    payload["event_type"] = str(payload.get("event_type", "misc"))
    payload["ts_event"] = payload.get("ts_event")
    payload["price"] = _to_float(payload.get("price"))
    payload["size"] = _to_float(payload.get("size"))
    payload["bar_index"] = _to_int(payload.get("bar_index"), -1)
    payload["side"] = str(payload.get("side", "flat"))
    payload["reason"] = str(payload.get("reason", ""))
    return payload


class EventReader:
    """Reader incremental de eventos de pivotes y órdenes/posiciones."""

    def __init__(self, pivot_log_path: Path, bot_events_path: Path, start_from_end: bool = False) -> None:
        self.pivot_reader = IncrementalFileReader(path=pivot_log_path, start_from_end=start_from_end)
        self.events_reader = IncrementalFileReader(path=bot_events_path, start_from_end=start_from_end)

    def read_updates(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        pivot_updates: list[dict[str, Any]] = []
        for line in self.pivot_reader.read_new_lines():
            parsed = parse_pivot_zone_line(line)
            if parsed is not None:
                pivot_updates.append(parsed)
            else:
                logger.debug("Línea de pivote no parseable: %s", line)

        bot_updates: list[dict[str, Any]] = []
        for line in self.events_reader.read_new_lines():
            parsed = parse_bot_event_line(line)
            if parsed is not None:
                bot_updates.append(parsed)
            else:
                logger.debug("Línea de evento no parseable: %s", line)

        return pivot_updates, bot_updates
