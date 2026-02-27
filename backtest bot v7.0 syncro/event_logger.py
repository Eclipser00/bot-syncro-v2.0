import json
import os
from datetime import datetime
from typing import Any, Dict


class EventLogger:
    """
    Logger ligero para escribir eventos en JSONL cumpliendo el esquema común.
    No modifica la lógica de trading; solo apendea líneas en outputs/*.jsonl.
    """

    def __init__(self, path: str, *, run_id: str | None = None, truncate: bool = False) -> None:
        self.path = path
        self.run_id = run_id or os.environ.get("RUN_ID") or ""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        truncate_flag = truncate or (os.environ.get("EVENT_LOG_TRUNCATE") == "1")
        if truncate_flag:
            try:
                with open(self.path, "w", encoding="utf-8") as f:
                    f.write("")
            except Exception:
                # No levantar excepciones para no interferir con el flujo
                pass

    def log(self, payload: Dict[str, Any]) -> None:
        # Sanitiza y completa campos mínimos
        event = {
            "run_id": payload.get("run_id") or self.run_id,
            "ts_event": payload.get("ts_event") or datetime.utcnow().isoformat() + "Z",
            "bar_index": int(payload.get("bar_index", -1)),
            "symbol": payload.get("symbol", "UNKNOWN"),
            "timeframe": payload.get("timeframe", "unknown"),
            "event_type": payload.get("event_type", "misc"),
            "side": payload.get("side", "flat"),
            "price_ref": payload.get("price_ref", "close"),
            "price": float(payload.get("price", 0.0) or 0.0),
            "size": float(payload.get("size", 0.0) or 0.0),
            "commission": float(payload.get("commission", 0.0) or 0.0),
            "slippage": float(payload.get("slippage", 0.0) or 0.0),
            "reason": payload.get("reason", ""),
            "order_id": str(payload.get("order_id", "")),
        }
        # Campos opcionales para instrumentación de paridad (zona pivote)
        for key in ("state_valid", "state_has_been_above", "state_has_been_below", "state_pivots"):
            if key in payload and payload.get(key) is not None:
                event[key] = int(payload.get(key))
        for key in ("state_mid", "state_width"):
            if key in payload and payload.get(key) is not None:
                event[key] = float(payload.get(key))
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            # No levantar excepciones para no interferir con el flujo de trading
            pass
