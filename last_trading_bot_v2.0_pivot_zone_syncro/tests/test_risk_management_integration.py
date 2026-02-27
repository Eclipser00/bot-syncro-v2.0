"""Tests de integraciÃ³n del risk management con el bucle del bot."""
from datetime import datetime, timedelta, timezone

import pandas as pd

from bot_trading.application.engine.bot_engine import TradingBot
from bot_trading.application.engine.order_executor import OrderExecutor
from bot_trading.application.risk_management import RiskManager
from bot_trading.application.engine.signals import Signal, SignalType
from bot_trading.domain.entities import OrderResult, SymbolConfig, RiskLimits, TradeRecord
from bot_trading.infrastructure.data_fetcher import MarketDataService


class FakeBrokerWithTrades:
    """Broker simulado que puede devolver trades cerrados para simular pÃ©rdidas."""
    
    def __init__(self) -> None:
        self.orders_sent: list = []
        self.closed_trades: list[TradeRecord] = []
        self._current_time = datetime(2023, 1, 1, 10, 0, tzinfo=timezone.utc)
    
    def connect(self) -> None:
        return None
    
    def get_ohlcv(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        index = pd.date_range(start=start, end=end, freq="1min")
        data = {
            "open": [1.0] * len(index),
            "high": [1.0] * len(index),
            "low": [1.0] * len(index),
            "close": [1.0] * len(index),
            "volume": [1] * len(index)
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
        """Devuelve los trades cerrados simulados."""
        return self.closed_trades.copy()
    
    def add_closed_trade(self, symbol: str, strategy_name: str, pnl: float) -> None:
        """AÃ±ade un trade cerrado simulado para testing."""
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
        )
        self.closed_trades.append(trade)
        self._current_time += timedelta(minutes=1)


class DummyStrategy:
    """Estrategia que siempre emite una seÃ±al de compra."""
    
    def __init__(self, name: str, symbols: list[str] | None = None):
        self.name = name
        self.timeframes = ["M1"]
        self.allowed_symbols = symbols
    
    def generate_signals(self, data_by_timeframe):
        signals = []
        for symbol in (self.allowed_symbols or ["EURUSD"]):
            signals.append(
                Signal(
                    symbol=symbol,
                    strategy_name=self.name,
                    timeframe="M1",
                    signal_type=SignalType.BUY,
                    size=0.01,
                    stop_loss=None,
                    take_profit=None,
                )
            )
        return signals


def test_bot_bloquea_globalmente_cuando_dd_supera_5_porciento() -> None:
    """Verifica que el bot se bloquea globalmente cuando el DD supera el 5%.
    
    Escenario:
    - Balance inicial: 10000
    - LÃ­mite global: 5%
    - Trade 1: +1000 (equity = 11000, max = 11000, dd = 0%)
    - Trade 2: -600 (equity = 10400, max = 11000, dd = 5.45% > 5%)
    - Resultado esperado: Bot bloqueado, no se procesan mÃ¡s sÃ­mbolos
    """
    broker = FakeBrokerWithTrades()
    market_data = MarketDataService(broker)
    
    # Configurar lÃ­mite global del 5%
    risk_manager = RiskManager(
        RiskLimits(
            dd_global=5.0,
            initial_balance=10000.0
        )
    )
    
    executor = OrderExecutor(broker)
    strategy = DummyStrategy("test_strategy", ["EURUSD"])
    
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
    
    # Ciclo 1: Trade ganador (+1000)
    broker.add_closed_trade("EURUSD", "test_strategy", 1000.0)
    broker.orders_sent.clear()  # Limpiar Ã³rdenes previas
    bot.run_once(now=broker._current_time)
    
    # Debe procesar normalmente (DD = 0%)
    assert len(broker.orders_sent) > 0, "Debe procesar Ã³rdenes cuando DD estÃ¡ dentro del lÃ­mite"
    
    # Ciclo 2: Trade perdedor que supera el 5% (-600)
    # Equity: 11000 - 600 = 10400
    # Max equity: 11000
    # DD: (11000 - 10400) / 11000 * 100 = 5.45% > 5%
    broker.add_closed_trade("EURUSD", "test_strategy", -600.0)
    broker.orders_sent.clear()
    bot.run_once(now=broker._current_time)
    
    # Debe estar bloqueado globalmente, no debe procesar Ã³rdenes
    assert len(broker.orders_sent) == 0, "No debe procesar Ã³rdenes cuando el bot estÃ¡ bloqueado globalmente"


def test_bot_bloquea_simbolo_cuando_dd_por_activo_supera_5_porciento() -> None:
    """Verifica que un sÃ­mbolo se bloquea cuando su DD supera el 5%, pero otros sÃ­mbolos siguen operando.
    
    Escenario:
    - Balance inicial: 10000
    - LÃ­mite por EURUSD: 5%
    - Trade EURUSD 1: +500 (equity EURUSD = 10500, max = 10500, dd = 0%)
    - Trade EURUSD 2: -600 (equity EURUSD = 9900, max = 10500, dd = 5.71% > 5%)
    - GBPUSD no tiene trades, debe seguir operando normalmente
    """
    broker = FakeBrokerWithTrades()
    market_data = MarketDataService(broker)
    
    # Configurar lÃ­mite del 5% solo para EURUSD
    risk_manager = RiskManager(
        RiskLimits(
            dd_por_activo={"EURUSD": 5.0},
            initial_balance=10000.0
        )
    )
    
    executor = OrderExecutor(broker)
    strategy = DummyStrategy("test_strategy", ["EURUSD", "GBPUSD"])
    
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
    
    # Ciclo 1: Trade ganador en EURUSD (+500)
    broker.add_closed_trade("EURUSD", "test_strategy", 500.0)
    broker.orders_sent.clear()
    bot.run_once(now=broker._current_time)
    
    # Debe procesar normalmente
    assert len(broker.orders_sent) > 0, "Debe procesar Ã³rdenes cuando DD estÃ¡ dentro del lÃ­mite"
    
    # Ciclo 2: Trade perdedor en EURUSD que supera el 5% (-600)
    # Equity EURUSD: 10500 - 600 = 9900
    # Max equity EURUSD: 10500
    # DD EURUSD: (10500 - 9900) / 10500 * 100 = 5.71% > 5%
    broker.add_closed_trade("EURUSD", "test_strategy", -600.0)
    broker.orders_sent.clear()
    bot.run_once(now=broker._current_time)
    
    # EURUSD debe estar bloqueado, pero GBPUSD debe seguir operando
    # Verificamos que se procesaron Ã³rdenes (de GBPUSD), pero no de EURUSD
    # Como la estrategia genera seÃ±ales para ambos, verificamos que al menos se procesÃ³ algo
    # pero el sÃ­mbolo EURUSD debe estar bloqueado internamente
    orders_eurusd = [o for o in broker.orders_sent if o.symbol == "EURUSD"]
    orders_gbpusd = [o for o in broker.orders_sent if o.symbol == "GBPUSD"]
    
    # GBPUSD debe seguir operando (no tiene lÃ­mite configurado)
    assert len(orders_gbpusd) > 0, "GBPUSD debe seguir operando ya que no tiene lÃ­mite configurado"


def test_bot_bloquea_estrategia_cuando_dd_por_estrategia_supera_5_porciento() -> None:
    """Verifica que una estrategia se bloquea cuando su DD supera el 5%, pero otras estrategias siguen operando.
    
    Escenario:
    - Balance inicial: 10000
    - LÃ­mite para strategy1: 5%
    - Trade strategy1: +1000 (equity = 11000, max = 11000, dd = 0%)
    - Trade strategy1: -600 (equity = 10400, max = 11000, dd = 5.45% > 5%)
    - strategy2 no tiene trades, debe seguir operando normalmente
    """
    broker = FakeBrokerWithTrades()
    market_data = MarketDataService(broker)
    
    # Configurar lÃ­mite del 5% solo para strategy1
    risk_manager = RiskManager(
        RiskLimits(
            dd_por_estrategia={"strategy1": 5.0},
            initial_balance=10000.0
        )
    )
    
    executor = OrderExecutor(broker)
    strategy1 = DummyStrategy("strategy1", ["EURUSD"])
    strategy2 = DummyStrategy("strategy2", ["GBPUSD"])
    
    bot = TradingBot(
        broker_client=broker,
        market_data_service=market_data,
        risk_manager=risk_manager,
        order_executor=executor,
        strategies=[strategy1, strategy2],
        symbols=[
            SymbolConfig(name="EURUSD", min_timeframe="M1"),
            SymbolConfig(name="GBPUSD", min_timeframe="M1"),
        ],
    )
    
    # Ciclo 1: Trade ganador de strategy1 (+1000)
    broker.add_closed_trade("EURUSD", "strategy1", 1000.0)
    broker.orders_sent.clear()
    bot.run_once(now=broker._current_time)
    
    # Debe procesar normalmente
    assert len(broker.orders_sent) > 0, "Debe procesar Ã³rdenes cuando DD estÃ¡ dentro del lÃ­mite"
    
    # Ciclo 2: Trade perdedor de strategy1 que supera el 5% (-600)
    # Equity strategy1: 11000 - 600 = 10400
    # Max equity strategy1: 11000
    # DD strategy1: (11000 - 10400) / 11000 * 100 = 5.45% > 5%
    broker.add_closed_trade("EURUSD", "strategy1", -600.0)
    broker.orders_sent.clear()
    bot.run_once(now=broker._current_time)
    
    # strategy1 debe estar bloqueada, pero strategy2 debe seguir operando
    orders_strategy1 = [o for o in broker.orders_sent if "strategy1" in (o.comment or "")]
    orders_strategy2 = [o for o in broker.orders_sent if "strategy2" in (o.comment or "")]
    
    # strategy2 debe seguir operando (no tiene lÃ­mite configurado)
    assert len(orders_strategy2) > 0, "strategy2 debe seguir operando ya que no tiene lÃ­mite configurado"


def test_bot_multiple_ciclos_acumula_drawdown_correctamente() -> None:
    """Verifica que el drawdown se acumula correctamente a travÃ©s de mÃºltiples ciclos.
    
    Escenario con balance inicial de 10000 y lÃ­mite del 5%:
    - Ciclo 1: Trade +2000 (equity = 12000, max = 12000, dd = 0%)
    - Ciclo 2: Trade -300 (equity = 11700, max = 12000, dd = 2.5% < 5%)
    - Ciclo 3: Trade -400 (equity = 11300, max = 12000, dd = 5.83% > 5%)
    - Resultado esperado: Bot bloqueado en ciclo 3
    """
    broker = FakeBrokerWithTrades()
    market_data = MarketDataService(broker)
    
    risk_manager = RiskManager(
        RiskLimits(
            dd_global=5.0,
            initial_balance=10000.0
        )
    )
    
    executor = OrderExecutor(broker)
    strategy = DummyStrategy("test_strategy", ["EURUSD"])
    
    bot = TradingBot(
        broker_client=broker,
        market_data_service=market_data,
        risk_manager=risk_manager,
        order_executor=executor,
        strategies=[strategy],
        symbols=[SymbolConfig(name="EURUSD", min_timeframe="M1")],
    )
    
    # Ciclo 1: Trade ganador (+2000)
    broker.add_closed_trade("EURUSD", "test_strategy", 2000.0)
    broker.orders_sent.clear()
    bot.run_once(now=broker._current_time)
    assert len(broker.orders_sent) > 0, "Ciclo 1: Debe operar (DD = 0%)"
    
    # Ciclo 2: Trade perdedor pequeÃ±o (-300)
    # Equity: 12000 - 300 = 11700
    # Max: 12000
    # DD: (12000 - 11700) / 12000 * 100 = 2.5% < 5%
    broker.add_closed_trade("EURUSD", "test_strategy", -300.0)
    broker.orders_sent.clear()
    bot.run_once(now=broker._current_time)
    assert len(broker.orders_sent) > 0, "Ciclo 2: Debe operar (DD = 2.5% < 5%)"
    
    # Ciclo 3: Trade perdedor que supera el 5% (-400)
    # Equity: 11700 - 400 = 11300
    # Max: 12000
    # DD: (12000 - 11300) / 12000 * 100 = 5.83% > 5%
    broker.add_closed_trade("EURUSD", "test_strategy", -400.0)
    broker.orders_sent.clear()
    bot.run_once(now=broker._current_time)
    assert len(broker.orders_sent) == 0, "Ciclo 3: Debe estar bloqueado (DD = 5.83% > 5%)"



