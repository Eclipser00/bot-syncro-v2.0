"""Contratos base para estrategias de trading."""
from __future__ import annotations

from typing import Protocol
import pandas as pd

from bot_trading.application.engine.signals import Signal


class Strategy(Protocol):
    """Protocolo que define la interfaz mínima de una estrategia."""

    name: str
    timeframes: list[str]

    def generate_signals(self, data_by_timeframe: dict[str, pd.DataFrame]) -> list[Signal]:
        """Genera señales basadas en los datos provistos."""
        raise NotImplementedError(
            "Debería analizar los dataframes y producir señales operables."
        )
