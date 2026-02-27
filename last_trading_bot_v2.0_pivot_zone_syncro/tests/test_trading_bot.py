"""Tests del motor principal del bot."""
from datetime import datetime, timedelta, timezone

import pandas as pd

from bot_trading.application.engine.bot_engine import TradingBot
from bot_trading.application.engine.order_executor import OrderExecutor
from bot_trading.application.risk_management import RiskManager
from bot_trading.application.strategies.base import Strategy
from bot_trading.application.engine.signals import Signal, SignalType
from bot_trading.domain.entities import OrderResult, RiskLimits, SymbolConfig, TradeRecord
from bot_trading.infrastructure.data_fetcher import MarketDataService


class FakeBroker:
    """ImplementaciÃ³n falsa de BrokerClient para pruebas de integraciÃ³n."""

    def __init__(self) -> None:
        self.orders_sent: list = []

    def connect(self) -> None:
        return None

    def get_ohlcv(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        index = pd.date_range(start=start, end=end, freq="1min")
        data = {"open": [1.0] * len(index), "high": [1.0] * len(index), "low": [1.0] * len(index), "close": [1.0] * len(index), "volume": [1] * len(index)}
        return pd.DataFrame(data, index=index)

    def send_market_order(self, order_request):
        self.orders_sent.append(order_request)
        return OrderResult(success=True, order_id=len(self.orders_sent))

    def get_open_positions(self):
        return []

    def get_closed_trades(self):
        return []


class DummyStrategy:
    """Estrategia que siempre emite una seÃ±al de compra."""

    name = "dummy"
    timeframes = ["M1"]

    def generate_signals(self, data_by_timeframe):
        return [
            Signal(
                symbol="EURUSD",
                strategy_name=self.name,
                timeframe="M1",
                signal_type=SignalType.BUY,
                size=0.01,
                stop_loss=None,
                take_profit=None,
            )
        ]


def test_trading_bot_run_once_ejecuta_flujo_basico() -> None:
    """run_once debe descargar datos, evaluar riesgo y enviar Ã³rdenes."""
    broker = FakeBroker()
    market_data = MarketDataService(broker)
    risk_manager = RiskManager(RiskLimits())
    executor = OrderExecutor(broker)
    bot = TradingBot(
        broker_client=broker,
        market_data_service=market_data,
        risk_manager=risk_manager,
        order_executor=executor,
        strategies=[DummyStrategy()],
        symbols=[SymbolConfig(name="EURUSD", min_timeframe="M1")],
    )

    now = datetime(2023, 1, 1, 0, 10)
    bot.run_once(now=now)

    assert len(broker.orders_sent) == 1


def test_trading_bot_run_once_invoca_market_data_callback() -> None:
    broker = FakeBroker()
    market_data = MarketDataService(broker)
    risk_manager = RiskManager(RiskLimits())
    executor = OrderExecutor(broker)
    callback_calls: list[tuple[str, set[str]]] = []

    def _on_market_data(symbol_cfg, data_by_timeframe, current_time) -> None:
        callback_calls.append((symbol_cfg.name, set(data_by_timeframe.keys())))

    bot = TradingBot(
        broker_client=broker,
        market_data_service=market_data,
        risk_manager=risk_manager,
        order_executor=executor,
        strategies=[DummyStrategy()],
        symbols=[SymbolConfig(name="EURUSD", min_timeframe="M1")],
        market_data_callback=_on_market_data,
    )

    now = datetime(2023, 1, 1, 0, 10)
    bot.run_once(now=now)

    assert callback_calls
    symbol_name, timeframes = callback_calls[0]
    assert symbol_name == "EURUSD"
    assert "M1" in timeframes


def test_update_trade_history_emite_cierre_desde_trade_history_con_precio_real() -> None:
    class BrokerWithClosedTrades(FakeBroker):
        def __init__(self, trades: list[TradeRecord]) -> None:
            super().__init__()
            self._closed_trades = trades

        def get_closed_trades(self):
            return list(self._closed_trades)

    class ExecutorSpy(OrderExecutor):
        def __init__(self, broker_client) -> None:
            super().__init__(broker_client)
            self.events: list[dict] = []

        def _emit_event(self, payload: dict) -> None:  # type: ignore[override]
            self.events.append(payload)

    trade = TradeRecord(
        symbol="EURUSD",
        strategy_name="dummy",
        entry_time=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        exit_time=datetime(2026, 1, 1, 10, 15, tzinfo=timezone.utc),
        entry_price=1.1000,
        exit_price=1.1055,
        size=0.20,
        pnl=110.0,
        stop_loss=1.0950,
        take_profit=1.1100,
    )

    broker = BrokerWithClosedTrades([trade])
    market_data = MarketDataService(broker)
    risk_manager = RiskManager(RiskLimits())
    executor = ExecutorSpy(broker)
    bot = TradingBot(
        broker_client=broker,
        market_data_service=market_data,
        risk_manager=risk_manager,
        order_executor=executor,
        strategies=[],
        symbols=[SymbolConfig(name="EURUSD", min_timeframe="M1")],
    )

    bot._update_trade_history()
    bot._update_trade_history()

    assert len(bot.trade_history) == 1
    assert len(executor.events) == 2

    fill_event = [e for e in executor.events if e.get("event_type") == "order_fill"][-1]
    position_event = [e for e in executor.events if e.get("event_type") == "position"][-1]

    assert fill_event["reason"] == "close_trade_history"
    assert position_event["reason"] == "position_close_trade_history"
    assert float(fill_event["price"]) == 1.1055
    assert float(position_event["price"]) == 1.1055

