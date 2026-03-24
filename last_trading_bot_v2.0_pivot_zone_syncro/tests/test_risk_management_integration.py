"""Tests de integracion del risk management con el bucle del bot."""
from datetime import datetime, timedelta, timezone

import pandas as pd

from bot_trading.application.engine.bot_engine import TradingBot
from bot_trading.application.engine.order_executor import OrderExecutor
from bot_trading.application.engine.signals import Signal, SignalType
from bot_trading.application.risk_management import RiskManager
from bot_trading.domain.entities import AccountInfo, OrderResult, RiskLimits, SymbolConfig, TradeRecord
from bot_trading.infrastructure.data_fetcher import MarketDataService


class FakeBrokerWithTrades:
    """Broker simulado que devuelve trades cerrados y balance actualizado."""

    def __init__(self, initial_balance: float = 10000.0) -> None:
        self.orders_sent: list = []
        self.closed_trades: list[TradeRecord] = []
        self._current_time = datetime(2023, 1, 1, 10, 0, tzinfo=timezone.utc)
        self._balance = float(initial_balance)

    def connect(self) -> None:
        return None

    def get_ohlcv(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        index = pd.date_range(start=start, end=end, freq="1min")
        data = {
            "open": [1.0] * len(index),
            "high": [1.0] * len(index),
            "low": [1.0] * len(index),
            "close": [1.0] * len(index),
            "volume": [1] * len(index),
        }
        df = pd.DataFrame(data, index=index)
        df.attrs["symbol"] = symbol
        return df

    def send_market_order(self, order_request):
        self.orders_sent.append(order_request)
        return OrderResult(success=True, order_id=len(self.orders_sent))

    def get_open_positions(self):
        return []

    def get_closed_trades(self):
        return self.closed_trades.copy()

    def get_account_info(self) -> AccountInfo:
        return AccountInfo(
            balance=self._balance,
            equity=self._balance,
            margin=0.0,
            margin_free=self._balance,
            margin_level=None,
        )

    def add_closed_trade(self, symbol: str, strategy_name: str, pnl: float) -> None:
        entry_time = self._current_time - timedelta(minutes=10)
        exit_time = self._current_time
        trade = TradeRecord(
            symbol=symbol,
            strategy_name=strategy_name,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=1.0,
            exit_price=1.0,
            size=0.01,
            pnl=pnl,
            stop_loss=None,
            take_profit=None,
            exit_deal_ticket=len(self.closed_trades) + 1,
        )
        self.closed_trades.append(trade)
        self._balance += pnl
        self._current_time += timedelta(minutes=1)


class DummyStrategy:
    """Estrategia que emite una señal de compra para el simbolo procesado."""

    def __init__(self, name: str):
        self.name = name
        self.timeframes = ["M1"]
        self.allowed_symbols = None

    def generate_signals(self, data_by_timeframe):
        df = data_by_timeframe["M1"]
        symbol = str(df.attrs.get("symbol", "EURUSD"))
        return [
            Signal(
                symbol=symbol,
                strategy_name=self.name,
                timeframe="M1",
                signal_type=SignalType.BUY,
                size=0.01,
                stop_loss=None,
                take_profit=None,
            )
        ]


def test_bot_bloquea_globalmente_cuando_dd_actual_supera_5_porciento(tmp_path) -> None:
    broker = FakeBrokerWithTrades()
    market_data = MarketDataService(broker)
    risk_manager = RiskManager(
        RiskLimits(
            dd_global=5.0,
            risk_state_path=str(tmp_path / "risk_state.json"),
        )
    )
    executor = OrderExecutor(broker)
    strategy = DummyStrategy("test_strategy")
    bot = TradingBot(
        broker_client=broker,
        market_data_service=market_data,
        risk_manager=risk_manager,
        order_executor=executor,
        strategies=[strategy],
        symbols=[
            SymbolConfig(name="EURUSD", min_timeframe="M1"),
            SymbolConfig(name="GBPUSD", min_timeframe="M1"),
        ],
    )

    broker.add_closed_trade("EURUSD", "test_strategy", 1000.0)
    broker.orders_sent.clear()
    bot.run_once(now=broker._current_time)
    assert len(broker.orders_sent) == 2

    broker.add_closed_trade("EURUSD", "test_strategy", -600.0)
    broker.orders_sent.clear()
    bot.run_once(now=broker._current_time)
    assert len(broker.orders_sent) == 0


def test_bot_bloquea_simbolo_cuando_dd_por_activo_supera_5_porciento(tmp_path) -> None:
    broker = FakeBrokerWithTrades()
    market_data = MarketDataService(broker)
    risk_manager = RiskManager(
        RiskLimits(
            dd_por_activo={"EURUSD": 5.0},
            risk_state_path=str(tmp_path / "risk_state.json"),
        )
    )
    executor = OrderExecutor(broker)
    strategy = DummyStrategy("test_strategy")
    bot = TradingBot(
        broker_client=broker,
        market_data_service=market_data,
        risk_manager=risk_manager,
        order_executor=executor,
        strategies=[strategy],
        symbols=[
            SymbolConfig(name="EURUSD", min_timeframe="M1"),
            SymbolConfig(name="GBPUSD", min_timeframe="M1"),
        ],
    )

    broker.add_closed_trade("EURUSD", "test_strategy", 500.0)
    broker.orders_sent.clear()
    bot.run_once(now=broker._current_time)
    assert {order.symbol for order in broker.orders_sent} == {"EURUSD", "GBPUSD"}

    broker.add_closed_trade("EURUSD", "test_strategy", -600.0)
    broker.orders_sent.clear()
    bot.run_once(now=broker._current_time)

    assert [order.symbol for order in broker.orders_sent] == ["GBPUSD"]


def test_bot_bloquea_estrategia_cuando_dd_por_estrategia_supera_5_porciento(tmp_path) -> None:
    broker = FakeBrokerWithTrades()
    market_data = MarketDataService(broker)
    risk_manager = RiskManager(
        RiskLimits(
            dd_por_estrategia={"strategy1": 5.0},
            risk_state_path=str(tmp_path / "risk_state.json"),
        )
    )
    executor = OrderExecutor(broker)
    strategy1 = DummyStrategy("strategy1")
    strategy2 = DummyStrategy("strategy2")
    bot = TradingBot(
        broker_client=broker,
        market_data_service=market_data,
        risk_manager=risk_manager,
        order_executor=executor,
        strategies=[strategy1, strategy2],
        symbols=[SymbolConfig(name="EURUSD", min_timeframe="M1")],
    )

    broker.add_closed_trade("EURUSD", "strategy1", 1000.0)
    broker.orders_sent.clear()
    bot.run_once(now=broker._current_time)
    assert len(broker.orders_sent) == 2

    broker.add_closed_trade("EURUSD", "strategy1", -600.0)
    broker.orders_sent.clear()
    bot.run_once(now=broker._current_time)

    comments = [order.comment for order in broker.orders_sent]
    assert comments == ["strategy2-M1"]


def test_bot_reanuda_operacion_tras_nuevo_trade_ganador(tmp_path) -> None:
    broker = FakeBrokerWithTrades()
    market_data = MarketDataService(broker)
    risk_manager = RiskManager(
        RiskLimits(
            dd_global=5.0,
            risk_state_path=str(tmp_path / "risk_state.json"),
        )
    )
    executor = OrderExecutor(broker)
    strategy = DummyStrategy("test_strategy")
    bot = TradingBot(
        broker_client=broker,
        market_data_service=market_data,
        risk_manager=risk_manager,
        order_executor=executor,
        strategies=[strategy],
        symbols=[SymbolConfig(name="EURUSD", min_timeframe="M1")],
    )

    broker.add_closed_trade("EURUSD", "test_strategy", 1000.0)
    bot.run_once(now=broker._current_time)

    broker.add_closed_trade("EURUSD", "test_strategy", -600.0)
    broker.orders_sent.clear()
    bot.run_once(now=broker._current_time)
    assert len(broker.orders_sent) == 0

    broker.add_closed_trade("EURUSD", "test_strategy", 300.0)
    broker.orders_sent.clear()
    bot.run_once(now=broker._current_time)
    assert len(broker.orders_sent) == 1
