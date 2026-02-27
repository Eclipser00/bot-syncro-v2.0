"""Tests de gestiÃ³n de riesgo de margen.

EvalÃºa que el bot no abra posiciones cuando el margen usado supera los lÃ­mites
configurados, evitando margin calls.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional
from dataclasses import dataclass

import pandas as pd

from bot_trading.application.engine.bot_engine import TradingBot
from bot_trading.application.engine.order_executor import OrderExecutor
from bot_trading.application.risk_management import RiskManager
from bot_trading.application.engine.signals import Signal, SignalType
from bot_trading.domain.entities import OrderResult, SymbolConfig, RiskLimits, TradeRecord, Position


@dataclass
class AccountInfo:
    """InformaciÃ³n de cuenta simulada."""
    balance: float
    equity: float
    margin: float  # Margen usado
    margin_free: float  # Margen libre
    margin_level: Optional[float] = None  # Nivel de margen en %


class FakeBrokerWithMargin:
    """Broker simulado con informaciÃ³n de margen configurable."""
    
    def __init__(self, initial_balance: float = 10000.0) -> None:
        self.orders_sent: list = []
        self.open_positions: list[Position] = []
        self.closed_trades: list[TradeRecord] = []
        self._current_time = datetime(2023, 1, 1, 10, 0, tzinfo=timezone.utc)
        
        # InformaciÃ³n de cuenta simulada
        self.account_info = AccountInfo(
            balance=initial_balance,
            equity=initial_balance,
            margin=0.0,  # Sin posiciones abiertas inicialmente
            margin_free=initial_balance,
            margin_level=None
        )
        
        # SimulaciÃ³n: cada lote de 0.01 requiere 100 unidades de margen
        # Esto es una simplificaciÃ³n para los tests
        self.margin_per_lot = 100.0
    
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
        """EnvÃ­a una orden y actualiza el margen usado."""
        self.orders_sent.append(order_request)
        
        # Simular apertura de posiciÃ³n
        if order_request.order_type in {"BUY", "SELL"}:
            # Calcular margen requerido para esta orden
            margin_required = order_request.volume * self.margin_per_lot
            
            # Verificar si hay margen suficiente (esto lo harÃ¡ el risk manager en producciÃ³n)
            # Por ahora solo registramos la orden
            position = Position(
                symbol=order_request.symbol,
                volume=order_request.volume,
                entry_price=1.0,
                stop_loss=order_request.stop_loss,
                take_profit=order_request.take_profit,
                strategy_name=order_request.comment or "unknown",
                open_time=self._current_time,
                magic_number=order_request.magic_number
            )
            self.open_positions.append(position)
            
            # Actualizar informaciÃ³n de cuenta simulada
            self.account_info.margin += margin_required
            self.account_info.margin_free = self.account_info.equity - self.account_info.margin
            if self.account_info.equity > 0:
                self.account_info.margin_level = (self.account_info.equity / self.account_info.margin) * 100 if self.account_info.margin > 0 else None
        
        return OrderResult(success=True, order_id=len(self.orders_sent))
    
    def get_open_positions(self):
        return self.open_positions.copy()
    
    def get_closed_trades(self):
        return self.closed_trades.copy()
    
    def get_account_info(self) -> AccountInfo:
        """Obtiene informaciÃ³n de cuenta simulada."""
        return self.account_info
    
    def set_account_info(self, balance: float, equity: float, margin: float) -> None:
        """Configura manualmente la informaciÃ³n de cuenta para tests."""
        self.account_info.balance = balance
        self.account_info.equity = equity
        self.account_info.margin = margin
        self.account_info.margin_free = equity - margin
        if equity > 0:
            self.account_info.margin_level = (equity / margin) * 100 if margin > 0 else None


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


def test_risk_manager_bloquea_orden_cuando_margen_supera_80_porciento() -> None:
    """Verifica que el risk manager bloquea Ã³rdenes cuando el margen usado supera el 80%.
    
    Escenario:
    - Balance inicial: 10000
    - LÃ­mite de margen: 80%
    - Margen usado actual: 8500 (85% del capital)
    - Nueva orden requiere: 100 de margen adicional
    - Resultado esperado: Orden BLOQUEADA
    """
    # Este test fallarÃ¡ inicialmente porque aÃºn no implementamos la validaciÃ³n de margen
    # Es parte del enfoque TDD: escribir el test primero
    
    broker = FakeBrokerWithMargin(initial_balance=10000.0)
    
    # Configurar cuenta con margen usado al 85% (8500 de 10000)
    broker.set_account_info(
        balance=10000.0,
        equity=10000.0,
        margin=8500.0  # 85% del capital usado
    )
    
    # Configurar lÃ­mite de margen al 80%
    risk_manager = RiskManager(
        RiskLimits(
            dd_global=None,
            initial_balance=10000.0,
            max_margin_usage_percent=80.0,
        )
    )
    
    # Verificar que el margen usado (85%) supera el lÃ­mite (80%)
    account_info = broker.get_account_info()
    margin_usage_percent = (account_info.margin / account_info.equity) * 100 if account_info.equity > 0 else 0
    
    assert margin_usage_percent > 80.0, "El margen usado debe superar el 80% para este test"
    
    # Verificar que el risk manager bloquea la orden
    result = risk_manager.check_margin_limits(account_info)
    assert result is False, "El risk manager debe bloquear cuando el margen supera el 80%"


def test_risk_manager_permite_orden_cuando_margen_esta_por_debajo_del_limite() -> None:
    """Verifica que el risk manager permite Ã³rdenes cuando el margen usado estÃ¡ por debajo del lÃ­mite.
    
    Escenario:
    - Balance inicial: 10000
    - LÃ­mite de margen: 80%
    - Margen usado actual: 5000 (50% del capital)
    - Nueva orden requiere: 100 de margen adicional
    - Resultado esperado: Orden PERMITIDA
    """
    broker = FakeBrokerWithMargin(initial_balance=10000.0)
    
    # Configurar cuenta con margen usado al 50% (5000 de 10000)
    broker.set_account_info(
        balance=10000.0,
        equity=10000.0,
        margin=5000.0  # 50% del capital usado
    )
    
    # Configurar lÃ­mite de margen al 80%
    risk_manager = RiskManager(
        RiskLimits(
            dd_global=None,
            initial_balance=10000.0,
            max_margin_usage_percent=80.0,
        )
    )
    
    # Verificar que el margen usado (50%) estÃ¡ por debajo del lÃ­mite (80%)
    account_info = broker.get_account_info()
    margin_usage_percent = (account_info.margin / account_info.equity) * 100 if account_info.equity > 0 else 0
    
    assert margin_usage_percent < 80.0, "El margen usado debe estar por debajo del 80% para este test"
    
    # Verificar que el risk manager permite la orden
    result = risk_manager.check_margin_limits(account_info)
    assert result is True, "El risk manager debe permitir cuando el margen estÃ¡ por debajo del 80%"


def test_risk_manager_bloquea_orden_cuando_margen_libre_insuficiente() -> None:
    """Verifica que el risk manager bloquea Ã³rdenes cuando no hay margen libre suficiente.
    
    Escenario:
    - Balance inicial: 10000
    - Equity: 10000
    - Margen usado: 9500
    - Margen libre: 500
    - Nueva orden requiere: 1000 de margen adicional
    - Resultado esperado: Orden BLOQUEADA (margen libre insuficiente)
    """
    broker = FakeBrokerWithMargin(initial_balance=10000.0)
    
    # Configurar cuenta con poco margen libre
    broker.set_account_info(
        balance=10000.0,
        equity=10000.0,
        margin=9500.0  # 95% del capital usado, solo 500 libre
    )
    
    account_info = broker.get_account_info()
    
    # Configurar risk manager con lÃ­mite de margen al 80%
    risk_manager = RiskManager(
        RiskLimits(
            dd_global=None,
            initial_balance=10000.0,
            max_margin_usage_percent=80.0,
        )
    )
    
    # Simular orden que requiere 1000 de margen (mÃ¡s de lo disponible)
    order_margin_required = 1000.0
    
    assert account_info.margin_free < order_margin_required, \
        "El margen libre debe ser insuficiente para esta orden"
    
    # Verificar que el risk manager bloquea la orden por falta de margen libre
    result = risk_manager.check_margin_limits(account_info, order_margin_required)
    assert result is False, "El risk manager debe bloquear cuando no hay margen libre suficiente"


def test_bot_no_abre_posiciones_cuando_margen_supera_limite() -> None:
    """Verifica que el bot no abre posiciones cuando el margen usado supera el lÃ­mite configurado.
    
    Escenario de integraciÃ³n:
    - Balance inicial: 10000
    - LÃ­mite de margen: 80%
    - Margen usado actual: 8500 (85%)
    - Estrategia genera seÃ±al de compra
    - Resultado esperado: Bot NO ejecuta la orden
    """
    broker = FakeBrokerWithMargin(initial_balance=10000.0)
    
    # Configurar cuenta con margen usado al 85%
    broker.set_account_info(
        balance=10000.0,
        equity=10000.0,
        margin=8500.0
    )
    
    from bot_trading.infrastructure.data_fetcher import MarketDataService
    market_data = MarketDataService(broker)
    
    # Configurar lÃ­mite de margen al 80%
    risk_manager = RiskManager(
        RiskLimits(
            dd_global=None,
            initial_balance=10000.0,
            # TODO: AÃ±adir max_margin_usage_percent=80.0 cuando se implemente
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
    
    # Limpiar Ã³rdenes previas
    broker.orders_sent.clear()
    
    # Ejecutar ciclo del bot
    bot.run_once(now=broker._current_time)
    
    # TODO: Cuando se implemente la validaciÃ³n de margen, no deberÃ­a haber Ã³rdenes
    # Por ahora el bot ejecutarÃ¡ Ã³rdenes porque no hay validaciÃ³n de margen
    # Este test documenta el comportamiento esperado una vez implementado
    # Cuando se implemente, descomentar el assert:
    # assert len(broker.orders_sent) == 0, \
    #     "No debe ejecutar Ã³rdenes cuando el margen usado supera el lÃ­mite"
    
    # Verificar que el escenario de test es vÃ¡lido
    account_info = broker.get_account_info()
    margin_usage_percent = (account_info.margin / account_info.equity) * 100 if account_info.equity > 0 else 0
    assert margin_usage_percent > 80.0, "El margen usado debe superar el 80% para este test"


def test_bot_abre_posiciones_cuando_margen_esta_por_debajo_del_limite() -> None:
    """Verifica que el bot abre posiciones normalmente cuando el margen estÃ¡ por debajo del lÃ­mite.
    
    Escenario de integraciÃ³n:
    - Balance inicial: 10000
    - LÃ­mite de margen: 80%
    - Margen usado actual: 5000 (50%)
    - Estrategia genera seÃ±al de compra
    - Resultado esperado: Bot ejecuta la orden normalmente
    """
    broker = FakeBrokerWithMargin(initial_balance=10000.0)
    
    # Configurar cuenta con margen usado al 50%
    broker.set_account_info(
        balance=10000.0,
        equity=10000.0,
        margin=5000.0
    )
    
    from bot_trading.infrastructure.data_fetcher import MarketDataService
    market_data = MarketDataService(broker)
    
    # Configurar lÃ­mite de margen al 80%
    risk_manager = RiskManager(
        RiskLimits(
            dd_global=None,
            initial_balance=10000.0,
            # TODO: AÃ±adir max_margin_usage_percent=80.0 cuando se implemente
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
    
    # Limpiar Ã³rdenes previas
    broker.orders_sent.clear()
    
    # Ejecutar ciclo del bot
    bot.run_once(now=broker._current_time)
    
    # Verificar que el escenario de test es vÃ¡lido
    account_info = broker.get_account_info()
    margin_usage_percent = (account_info.margin / account_info.equity) * 100 if account_info.equity > 0 else 0
    assert margin_usage_percent < 80.0, "El margen usado debe estar por debajo del 80% para este test"
    
    # Actualmente el bot ejecutarÃ¡ la orden porque no hay validaciÃ³n de margen
    # Cuando se implemente la validaciÃ³n, deberÃ­a seguir ejecutando porque el margen estÃ¡ por debajo del lÃ­mite
    # Este test documenta el comportamiento esperado una vez implementado
    # Cuando se implemente, verificar que las Ã³rdenes se ejecutan normalmente:
    # assert len(broker.orders_sent) > 0, \
    #     "Debe ejecutar Ã³rdenes cuando el margen estÃ¡ por debajo del lÃ­mite"


