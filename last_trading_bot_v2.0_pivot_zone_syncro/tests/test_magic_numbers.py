"""Tests de magic numbers y coexistencia de estrategias usando PivotZoneTestStrategy."""
import logging
from datetime import datetime, timezone

import pandas as pd

from bot_trading.main import FakeBroker
from bot_trading.application.engine.bot_engine import TradingBot
from bot_trading.application.engine.order_executor import OrderExecutor
from bot_trading.application.engine.signals import Signal, SignalType
from bot_trading.application.risk_management import RiskManager
from bot_trading.application.strategies.pivot_zone_test_strategy import PivotZoneTestStrategy
from bot_trading.domain.entities import RiskLimits, SymbolConfig
from bot_trading.infrastructure.data_fetcher import MarketDataService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyPivotStrategy(PivotZoneTestStrategy):
    """Subclase de PivotZoneTestStrategy que emite una BUY solo la primera vez."""

    def __init__(self, name: str, broker_client: FakeBroker):
        super().__init__(name=name, timeframes=[], broker_client=broker_client)
        self._emitted_once = False

    def generate_signals(self, data_by_timeframe: dict[str, pd.DataFrame]) -> list[Signal]:
        symbol = next(iter(data_by_timeframe.values())).attrs.get("symbol", "EURUSD")
        price_series = data_by_timeframe[self.tf_entry]["close"]
        entry = float(price_series.iloc[-1])

        if self._emitted_once:
            return []
        self._emitted_once = True

        stop = entry - 0.001
        tp = entry + 0.001
        return [
            Signal(
                symbol=symbol,
                strategy_name=self.name,
                timeframe=self.tf_entry,
                signal_type=SignalType.BUY,
                size=0.01,
                stop_loss=stop,
                take_profit=tp,
            )
        ]


def _build_bot(strategy: PivotZoneTestStrategy) -> tuple[TradingBot, FakeBroker]:
    broker = strategy.broker_client  # type: ignore[assignment]
    assert isinstance(broker, FakeBroker)
    market_data_service = MarketDataService(broker)
    risk_manager = RiskManager(RiskLimits(dd_global=1000))
    order_executor = OrderExecutor(broker)
    symbols = [SymbolConfig(name="EURUSD", min_timeframe="M1")]

    bot = TradingBot(
        broker_client=broker,
        market_data_service=market_data_service,
        risk_manager=risk_manager,
        order_executor=order_executor,
        strategies=[strategy],
        symbols=symbols,
    )
    return bot, broker


def test_no_duplicate_positions():
    """Verifica que no se abran mÃºltiples posiciones con el mismo Magic Number."""
    broker = FakeBroker()
    strategy = DummyPivotStrategy(name="test_strategy", broker_client=broker)
    bot, broker = _build_bot(strategy)

    # Primera ejecuciÃ³n - deberÃ­a abrir una posiciÃ³n
    bot.run_once(now=datetime.now(timezone.utc))
    first_orders = len(broker.orders_sent)
    first_positions = len(broker.open_positions)

    # Segunda ejecuciÃ³n - no deberÃ­a abrir otra posiciÃ³n
    bot.run_once(now=datetime.now(timezone.utc))
    second_orders = len(broker.orders_sent)
    second_positions = len(broker.open_positions)

    assert first_orders == 1
    assert first_positions == 1
    assert second_orders == first_orders
    assert second_positions == first_positions


def test_multiple_strategies_same_symbol():
    """Verifica que mÃºltiples estrategias puedan operar el mismo sÃ­mbolo sin interferencias."""
    broker = FakeBroker()
    strategy1 = DummyPivotStrategy(name="strategy_1", broker_client=broker)
    strategy2 = DummyPivotStrategy(name="strategy_2", broker_client=broker)
    market_data_service = MarketDataService(broker)
    risk_manager = RiskManager(RiskLimits(dd_global=1000))
    order_executor = OrderExecutor(broker)
    symbols = [SymbolConfig(name="EURUSD", min_timeframe="M1")]

    bot = TradingBot(
        broker_client=broker,
        market_data_service=market_data_service,
        risk_manager=risk_manager,
        order_executor=order_executor,
        strategies=[strategy1, strategy2],
        symbols=symbols,
    )

    bot.run_once(now=datetime.now(timezone.utc))

    orders = len(broker.orders_sent)
    positions = len(broker.open_positions)

    assert positions == 2
    assert orders == 2

    magic_numbers = [p.magic_number for p in broker.open_positions]
    assert len(set(magic_numbers)) == 2

