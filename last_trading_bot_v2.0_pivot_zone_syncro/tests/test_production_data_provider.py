from datetime import datetime, timedelta, timezone

import pandas as pd

from bot_trading.domain.entities import SymbolConfig
from bot_trading.infrastructure.data_fetcher import ProductionDataProvider


class _DummyBroker:
    """Broker simulado que devuelve series crecientes para verificar el buffer."""

    def __init__(self) -> None:
        self.calls = 0
        self.base_start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def get_ohlcv(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        self.calls += 1
        periods = 5 + self.calls  # cada llamada aÃ±ade una barra nueva
        idx = pd.date_range(self.base_start, periods=periods, freq="1min")
        data = {
            "open": [1.0] * periods,
            "high": [1.0] * periods,
            "low": [1.0] * periods,
            "close": list(range(periods)),
            "volume": [1] * periods,
        }
        df = pd.DataFrame(data, index=idx)
        df.attrs["symbol"] = symbol
        return df


def test_production_provider_accumulates_incremental_bars():
    broker = _DummyBroker()
    provider = ProductionDataProvider(
        broker_client=broker,
        lookback_days_entry=1,
        lookback_days_zone=1,
        lookback_days_stop=1,
    )
    symbol = SymbolConfig(name="EURUSD", min_timeframe="M1")

    now = datetime(2024, 1, 1, 0, 10, tzinfo=timezone.utc)
    data_first = provider.get_data(symbol, ["M1", "M5"], now)
    len_first = len(data_first["M1"])

    now_later = now + timedelta(minutes=1)
    data_second = provider.get_data(symbol, ["M1", "M5"], now_later)
    len_second = len(data_second["M1"])

    assert len_second > len_first, "El buffer debe crecer con nuevas velas"
    assert data_second["M1"].attrs.get("symbol") == "EURUSD"
    # El resample no debe devolver mÃ¡s velas que el timeframe base
    assert len(data_second["M5"]) <= len_second

