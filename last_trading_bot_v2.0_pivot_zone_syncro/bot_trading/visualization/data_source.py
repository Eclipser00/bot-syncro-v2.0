"""Fuentes de velas para el visualizador (development y production)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from bot_trading.domain.entities import SymbolConfig
from bot_trading.infrastructure.data_fetcher import (
    DevelopmentCsvDataProvider,
    MarketDataService,
    ProductionDataProvider,
)
from bot_trading.infrastructure.mt5_client import MetaTrader5Client

logger = logging.getLogger(__name__)

GetSharedDataFn = Callable[[SymbolConfig, list[str], datetime], dict[str, pd.DataFrame]]


class BaseCandleDataSource:
    """Contrato simple de ingesta incremental de velas."""

    def fetch_cycle(self) -> tuple[dict[str, pd.DataFrame], bool]:
        raise NotImplementedError

    def close(self) -> None:
        """Hook opcional para limpiar recursos externos."""
        return None


@dataclass
class DevelopmentCandleDataSource(BaseCandleDataSource):
    """Lee CSV M1 y proyecta/entrega velas M3 hasta agotar los archivos."""

    symbols: list[SymbolConfig]
    data_dir: Path
    base_timeframe: str
    lookback_days_entry: int
    lookback_days_zone: int
    lookback_days_stop: int

    def __post_init__(self) -> None:
        self.provider = DevelopmentCsvDataProvider(
            data_dir=self.data_dir,
            base_timeframe=self.base_timeframe,
            lookback_days_entry=self.lookback_days_entry,
            lookback_days_zone=self.lookback_days_zone,
            lookback_days_stop=self.lookback_days_stop,
        )
        self.service = MarketDataService(
            data_provider=self.provider,
            lookback_days_entry=self.lookback_days_entry,
            lookback_days_zone=self.lookback_days_zone,
            lookback_days_stop=self.lookback_days_stop,
        )
        self._exhausted_symbols: set[str] = set()

    def fetch_cycle(self) -> tuple[dict[str, pd.DataFrame], bool]:
        candles_by_symbol: dict[str, pd.DataFrame] = {}

        for symbol in self.symbols:
            if symbol.name in self._exhausted_symbols:
                continue

            simulated_now = self.provider.get_simulated_now([symbol]) or datetime.now(timezone.utc)
            try:
                data = self.service.get_data(
                    symbol=symbol,
                    target_timeframes=["M3", symbol.min_timeframe],
                    now=simulated_now,
                )
            except StopIteration:
                self._exhausted_symbols.add(symbol.name)
                logger.info("CSV agotado para %s", symbol.name)
                continue
            except Exception as exc:
                logger.exception("Error leyendo CSV para %s: %s", symbol.name, exc)
                continue

            m3 = data.get("M3")
            if m3 is not None and not m3.empty:
                candles_by_symbol[symbol.name] = m3

        finished = len(self._exhausted_symbols) == len(self.symbols)
        return candles_by_symbol, finished


@dataclass
class ProductionCandleDataSource(BaseCandleDataSource):
    """Obtiene velas M3 en producción.

    Prioridad:
    1) `shared_data_getter` (si el visualizador está integrado al loop del bot).
    2) `MarketDataService` + MT5 directo (fallback desacoplado).
    """

    symbols: list[SymbolConfig]
    lookback_days_entry: int
    lookback_days_zone: int
    lookback_days_stop: int
    max_retries: int
    retry_delay: float
    shared_data_getter: GetSharedDataFn | None = None

    def __post_init__(self) -> None:
        self._broker: MetaTrader5Client | None = None
        self._service: MarketDataService | None = None
        if self.shared_data_getter is not None:
            return

        self._broker = MetaTrader5Client(max_retries=self.max_retries, retry_delay=self.retry_delay)
        self._broker.connect()
        provider = ProductionDataProvider(
            broker_client=self._broker,
            lookback_days_entry=self.lookback_days_entry,
            lookback_days_zone=self.lookback_days_zone,
            lookback_days_stop=self.lookback_days_stop,
        )
        self._service = MarketDataService(
            data_provider=provider,
            lookback_days_entry=self.lookback_days_entry,
            lookback_days_zone=self.lookback_days_zone,
            lookback_days_stop=self.lookback_days_stop,
        )

    def fetch_cycle(self) -> tuple[dict[str, pd.DataFrame], bool]:
        candles_by_symbol: dict[str, pd.DataFrame] = {}
        now = datetime.now(timezone.utc)

        for symbol in self.symbols:
            try:
                if self.shared_data_getter is not None:
                    data = self.shared_data_getter(symbol, ["M3", symbol.min_timeframe], now)
                else:
                    if self._service is None:
                        raise RuntimeError("MarketDataService no inicializado en ProductionCandleDataSource")
                    data = self._service.get_data(
                        symbol=symbol,
                        target_timeframes=["M3", symbol.min_timeframe],
                        now=now,
                    )
            except Exception as exc:
                logger.exception("Error obteniendo velas de producción para %s: %s", symbol.name, exc)
                continue

            m3 = data.get("M3")
            if m3 is not None and not m3.empty:
                candles_by_symbol[symbol.name] = m3

        return candles_by_symbol, False

    def close(self) -> None:
        # MetaTrader5Client cierra conexión al destruirse; aquí solo liberamos referencia.
        self._service = None
        self._broker = None

