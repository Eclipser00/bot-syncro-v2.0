"""Tests del servicio de datos de mercado."""
from datetime import datetime, timedelta

import pandas as pd

from bot_trading.domain.entities import SymbolConfig
from bot_trading.infrastructure.data_fetcher import MarketDataService


class FakeBroker:
    """Broker simulado que devuelve datos secuenciales."""

    def get_ohlcv(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:  # noqa: D401,E501
        index = pd.date_range(start=start, end=end, freq="1min")
        data = {
            "open": range(len(index)),
            "high": range(len(index)),
            "low": range(len(index)),
            "close": range(len(index)),
            "volume": [1] * len(index),
        }
        return pd.DataFrame(data, index=index)


def test_market_data_service_resamplea_timeframes_correctamente() -> None:
    """Debe resamplear datos al conjunto de timeframes requerido."""
    start = datetime(2023, 1, 1, 0, 0)
    end = start + timedelta(minutes=9)
    broker = FakeBroker()
    service = MarketDataService(broker)
    symbol = SymbolConfig(name="EURUSD", min_timeframe="M1")

    result = service.get_resampled_data(symbol, ["M1", "M5"], start, end)

    assert set(result.keys()) == {"M1", "M5"}
    assert len(result["M1"]) == 10
    # Con label/closed=right se genera una vela parcial inicial si el rango inicia en el borde
    # del timeframe, por eso el resample devuelve 3 velas.
    assert len(result["M5"]) == 3

