from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from config import ACTIVE_ENV, ENVIRONMENTS, settings
from bot_trading.application.engine.bot_engine import TradingBot
from bot_trading.domain.entities import SymbolConfig
from bot_trading.infrastructure.data_fetcher import DevelopmentCsvDataProvider, MarketDataService


def _write_symbol_csv(csv_path, periods: int = 5) -> None:
    idx = pd.date_range(start="2024-01-01 00:00:00", periods=periods, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "datetime": idx.astype(str),
            "open": list(range(1, periods + 1)),
            "high": list(range(1, periods + 1)),
            "low": list(range(1, periods + 1)),
            "close": list(range(1, periods + 1)),
            "volume": [1.0] * periods,
        }
    )
    df.to_csv(csv_path, index=False)


def test_development_csv_provider_advances_and_exhausts(tmp_path) -> None:
    _write_symbol_csv(tmp_path / "EURUSD.csv", periods=3)
    provider = DevelopmentCsvDataProvider(
        data_dir=tmp_path,
        base_timeframe="M1",
        lookback_days_entry=0,
        lookback_days_zone=0,
        lookback_days_stop=0,
    )
    symbol = SymbolConfig(name="EURUSD", min_timeframe="M1")

    t1 = provider.get_simulated_now([symbol])
    assert t1 is not None
    data1 = provider.get_data(symbol, ["M1"], now=t1)
    assert len(data1["M1"]) == 1
    assert not data1["M1"]["close"].isna().any()

    t2 = provider.get_simulated_now([symbol])
    assert t2 is not None
    data2 = provider.get_data(symbol, ["M1"], now=t2)
    assert len(data2["M1"]) == 2
    assert not data2["M1"]["close"].isna().any()
    assert t2 > t1

    t3 = provider.get_simulated_now([symbol])
    assert t3 is not None
    data3 = provider.get_data(symbol, ["M1"], now=t3)
    assert len(data3["M1"]) == 3
    assert not data3["M1"]["close"].isna().any()

    with pytest.raises(StopIteration):
        provider.get_data(symbol, ["M1"], now=t3)


def test_development_csv_provider_does_not_start_at_end_when_lookback_exceeds_csv(tmp_path) -> None:
    _write_symbol_csv(tmp_path / "EURUSD.csv", periods=10)
    provider = DevelopmentCsvDataProvider(
        data_dir=tmp_path,
        base_timeframe="M1",
        lookback_days_entry=14,
        lookback_days_zone=14,
        lookback_days_stop=14,
    )
    symbol = SymbolConfig(name="EURUSD", min_timeframe="M1")

    t1 = provider.get_simulated_now([symbol])
    assert t1 is not None
    data1 = provider.get_data(symbol, ["M1"], now=t1)
    assert len(data1["M1"]) == 1
    assert not data1["M1"]["close"].isna().any()

    t2 = provider.get_simulated_now([symbol])
    assert t2 is not None
    data2 = provider.get_data(symbol, ["M1"], now=t2)
    assert len(data2["M1"]) == 2
    assert not data2["M1"]["close"].isna().any()
    assert t2 > t1


def test_development_csv_provider_resamples_m3_after_enough_bars(tmp_path) -> None:
    _write_symbol_csv(tmp_path / "EURUSD.csv", periods=10)
    provider = DevelopmentCsvDataProvider(
        data_dir=tmp_path,
        base_timeframe="M1",
        lookback_days_entry=0,
        lookback_days_zone=0,
        lookback_days_stop=0,
    )
    symbol = SymbolConfig(name="EURUSD", min_timeframe="M1")

    # Avanzar unas cuantas velas para que haya al menos 2 barras de M3.
    for _ in range(7):
        now = provider.get_simulated_now([symbol])
        assert now is not None
        data = provider.get_data(symbol, ["M1", "M3"], now=now)

    assert "M3" in data
    assert len(data["M3"]) >= 2


def test_trading_bot_run_synchronized_uses_csv_clock_without_sleep(tmp_path, monkeypatch) -> None:
    _write_symbol_csv(tmp_path / "EURUSD.csv", periods=4)
    provider = DevelopmentCsvDataProvider(
        data_dir=tmp_path,
        base_timeframe="M1",
        lookback_days_entry=0,
        lookback_days_zone=0,
        lookback_days_stop=0,
    )
    service = MarketDataService(provider)
    symbol = SymbolConfig(name="EURUSD", min_timeframe="M1")

    import time as _time

    monkeypatch.setattr(
        _time,
        "sleep",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("sleep llamado")),
    )

    class _Bot(TradingBot):
        def __init__(self, market_data_service):
            super().__init__(
                broker_client=object(),
                market_data_service=market_data_service,
                risk_manager=object(),
                order_executor=object(),
                strategies=[],
                symbols=[symbol],
            )
            self.times: list[datetime] = []

        def run_once(self, now: datetime | None = None) -> None:
            assert now is not None
            self.market_data_service.get_data(
                symbol=symbol,
                target_timeframes=["M1"],
                now=now,
            )
            self.times.append(now)

    bot = _Bot(service)
    bot.run_synchronized(timeframe_minutes=1, wait_after_close=5, skip_sleep_when_simulated=True)

    assert len(bot.times) == 4
    assert bot.times[1] > bot.times[0]
    assert bot.times[1] - bot.times[0] == timedelta(minutes=1)


def test_trading_bot_run_synchronized_respects_skip_flag_false(tmp_path, monkeypatch) -> None:
    _write_symbol_csv(tmp_path / "EURUSD.csv", periods=3)
    provider = DevelopmentCsvDataProvider(
        data_dir=tmp_path,
        base_timeframe="M1",
        lookback_days_entry=0,
        lookback_days_zone=0,
        lookback_days_stop=0,
    )
    service = MarketDataService(provider)
    symbol = SymbolConfig(name="EURUSD", min_timeframe="M1")

    class _Bot(TradingBot):
        def __init__(self, market_data_service):
            super().__init__(
                broker_client=object(),
                market_data_service=market_data_service,
                risk_manager=object(),
                order_executor=object(),
                strategies=[],
                symbols=[symbol],
            )
            self._fixed_now = datetime(2024, 1, 1, 0, 0, 50, tzinfo=timezone.utc)

        def _now(self) -> datetime:
            return self._fixed_now

        def run_once(self, now: datetime | None = None) -> None:
            # No llegaremos aqui en la prueba porque cortamos el sleep antes,
            # pero mantenemos la firma para coherencia.
            return None

    bot = _Bot(service)

    sleep_calls: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        raise SystemExit  # cortar el bucle tras el primer sleep

    monkeypatch.setattr(time, "sleep", _fake_sleep)

    with pytest.raises(SystemExit):
        bot.run_synchronized(timeframe_minutes=1, wait_after_close=0, skip_sleep_when_simulated=False)

    assert sleep_calls, "Debe haberse llamado a sleep cuando skip_sleep_when_simulated=False"
    # Con _fixed_now en segundo 50 y timeframe M1, deberian ser ~10s.
    assert 8.0 <= sleep_calls[0] <= 12.0


def test_development_config_skips_sleep_by_default() -> None:
    dev_loop = ENVIRONMENTS["development"].loop
    assert dev_loop.skip_sleep_when_simulated is True
    # settings siempre debe reflejar el entorno activo configurado en ACTIVE_ENV.
    assert settings.loop.skip_sleep_when_simulated is ENVIRONMENTS[ACTIVE_ENV].loop.skip_sleep_when_simulated

