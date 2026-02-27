"""Modelos de señales generadas por las estrategias."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Tipos de señales que pueden emitirse."""

    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"
    HOLD = "HOLD"


@dataclass
class Signal:
    """Señal producida por una estrategia."""

    symbol: str
    strategy_name: str
    timeframe: str
    signal_type: SignalType
    size: float
    stop_loss: float | None
    take_profit: float | None
