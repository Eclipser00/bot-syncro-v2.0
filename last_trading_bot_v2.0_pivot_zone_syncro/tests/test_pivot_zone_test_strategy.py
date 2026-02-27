import pandas as pd
import pytest
from types import SimpleNamespace

from bot_trading.application.strategies.pivot_zone_test_strategy import (
    PivotZoneTestStrategy,
    _Pivot3CandleState,
)
from bot_trading.application.engine.signals import SignalType
from bot_trading.domain.entities import AccountInfo, Position


def make_df(closes, highs=None, lows=None, symbol="EURUSD"):
    highs = highs or closes
    lows = lows or closes
    index = pd.date_range("2024-01-01", periods=len(closes), freq="1min")
    df = pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1] * len(closes),
        },
        index=index,
    )
    df.attrs["symbol"] = symbol
    return df


class FakeBrokerClient:
    def __init__(self, equity: float = 10_000.0) -> None:
        self.positions: list[Position] = []
        self.pending_orders = []
        self._next_order_id = 1
        self.account = AccountInfo(
            balance=equity,
            equity=equity,
            margin=0.0,
            margin_free=equity,
            margin_level=None,
        )
        self.symbol_info = SimpleNamespace(
            trade_tick_value=1.0,
            trade_tick_size=0.0001,
            volume_min=0.01,
            volume_step=0.01,
            volume_max=10.0,
        )

    def get_open_positions(self):
        return list(self.positions)

    def get_open_orders(self, symbol=None, magic_number=None):
        def _match(po):
            if symbol is not None and po.symbol != symbol:
                return False
            if magic_number is not None and po.magic_number != magic_number:
                return False
            return True

        return [po for po in self.pending_orders if _match(po)]

    def create_pending_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: float,
        stop_loss=None,
        take_profit=None,
        magic_number=None,
        comment=None,
    ):
        order_id = self._next_order_id
        self._next_order_id += 1
        po = SimpleNamespace(
            order_id=order_id,
            symbol=symbol,
            order_type=order_type,
            volume=volume,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            magic_number=magic_number,
            comment=comment,
        )
        self.pending_orders.append(po)
        return SimpleNamespace(success=True, order_id=order_id)

    def cancel_order(self, order_id: int) -> bool:
        before = len(self.pending_orders)
        self.pending_orders = [po for po in self.pending_orders if po.order_id != order_id]
        return len(self.pending_orders) < before

    def modify_order(self, order_id: int, price=None, stop_loss=None, take_profit=None):
        for po in self.pending_orders:
            if po.order_id == order_id:
                if price is not None:
                    po.price = price
                if stop_loss is not None:
                    po.stop_loss = stop_loss
                if take_profit is not None:
                    po.take_profit = take_profit
                return SimpleNamespace(success=True, order_id=order_id)
        return SimpleNamespace(success=False, order_id=None, error_message="not_found")

    def get_account_info(self):
        return self.account

    def _get_symbol_info(self, symbol: str):
        return self.symbol_info


class MetaTrader5Client(FakeBrokerClient):
    """Doble de pruebas con mismo nombre de clase que el broker real.

    La estrategia desactiva pendientes opuestas cuando detecta esta clase.
    """

    def modify_position_sl_tp(self, symbol: str, magic_number=None, stop_loss=None, take_profit=None):
        return SimpleNamespace(success=True, order_id=1)


def build_breakout_context(symbol_params=None):
    broker = FakeBrokerClient()
    strategy = PivotZoneTestStrategy(
        name="PivotZoneTest",
        broker_client=broker,
        tf_entry="M1",
        tf_zone="M5",
        tf_stop="M5",
        n1=3,
        n2=100,
        n3=2,
        size_pct=0.05,
        p=0.5,
        symbol_params=symbol_params or {},
    )
    symbol = "EURUSD"
    strategy._ensure_symbol_state(symbol)
    strategy._saved_zones_by_symbol[symbol] = [
        {"top": 101.0, "bot": 99.0, "bar": 0},
        {"top": 111.0, "bot": 109.0, "bar": 10},
    ]
    strategy._stop_last_min[symbol] = 99.0
    strategy._stop_pivot_mins[symbol] = [99.0]

    entry_df = make_df(
        closes=[99.5, 100.0, 100.6, 100.9, 101.2, 101.6],
        highs=[99.7, 100.2, 100.8, 101.1, 101.4, 101.8],
        lows=[99.3, 99.8, 100.4, 100.7, 100.8, 101.2],
        symbol=symbol,
    )
    zone_df = make_df(
        closes=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        highs=[100.0] * 6,
        lows=[100.0] * 6,
        symbol=symbol,
    )
    return strategy, broker, {"M1": entry_df, "M5": zone_df}


def test_pivot3candle_confirmations():
    state = _Pivot3CandleState()
    pmx, pmn = state.update(1.0, 3.0)
    assert pmx is None and pmn is None

    pmx, pmn = state.update(3.0, 2.0)
    assert pmx is None and pmn is None

    pmx, pmn = state.update(2.0, 1.0)
    assert pmx is None and pmn is None

    pmx, pmn = state.update(2.0, 0.5)
    assert pmx == pytest.approx(3.0)
    assert pmn is None

    pmx, pmn = state.update(2.0, 0.9)
    assert pmx is None and pmn is None

    pmx, pmn = state.update(2.0, 1.1)
    assert pmx is None
    assert pmn == pytest.approx(0.5)


def test_zone_lock_and_persistence():
    broker = FakeBrokerClient()
    strategy = PivotZoneTestStrategy(
        name="PivotZoneTest",
        broker_client=broker,
        tf_entry="M1",
        tf_zone="M1",
        tf_stop="M1",
        n1=3,
        n2=100,
        n3=2,
    )
    symbol = "EURUSD"
    strategy._ensure_symbol_state(symbol)
    proj = strategy._zone_project_by_symbol[symbol]
    proj.add_pivot(100.0)
    proj.add_pivot(99.5)
    proj.has_been_above = True
    proj.has_been_below = True
    # Ancho amplio para no resetear proyecto
    strategy._atr_series = lambda df, _n1: [3.0] * len(df)
    params = strategy._resolve_params(symbol)

    closes = [100.0] * 12 + [101.0, 99.0, 100.0, 100.0]
    highs = [100.1] * 12 + [101.0, 99.0, 100.0, 100.0]
    lows = [99.9] * 12 + [100.0, 98.0, 99.0, 99.0]
    df_zone = make_df(closes=closes, highs=highs, lows=lows, symbol=symbol)
    strategy._process_zone_tf(symbol, strategy._closed_series(df_zone), params)

    assert len(strategy._saved_zones_by_symbol[symbol]) == 1
    zone = strategy._saved_zones_by_symbol[symbol][0]
    assert zone["top"] > zone["bot"]
    proj_after = strategy._zone_project_by_symbol[symbol]
    assert proj_after.locked is False
    assert len(proj_after.pivots) <= 2


def test_zone_reject_too_close():
    broker = FakeBrokerClient()
    strategy = PivotZoneTestStrategy(
        name="PivotZoneTest",
        broker_client=broker,
        tf_entry="M1",
        tf_zone="M1",
        tf_stop="M1",
        n1=3,
        n2=100,
        n3=2,
    )
    symbol = "EURUSD"
    strategy._ensure_symbol_state(symbol)
    strategy._saved_zones_by_symbol[symbol] = [{"top": 100.2, "bot": 99.8, "bar": 0}]
    proj = strategy._zone_project_by_symbol[symbol]
    proj.pivots = [100.1, 99.9]
    proj.mid = 100.0
    proj.has_been_above = True
    proj.has_been_below = True

    df_zone = make_df(
        closes=[100.0, 100.0, 100.0],
        highs=[100.1, 100.1, 100.1],
        lows=[99.9, 99.9, 99.9],
        symbol=symbol,
    )
    params = strategy._resolve_params(symbol)
    strategy._process_zone_tf(symbol, strategy._closed_series(df_zone), params)
    assert len(strategy._saved_zones_by_symbol[symbol]) == 1


def test_breakout_long_requires_origin_filter():
    broker = FakeBrokerClient()
    strategy = PivotZoneTestStrategy(
        name="PivotZoneTest",
        broker_client=broker,
        tf_entry="M1",
        tf_zone="M5",
        tf_stop="M5",
        n1=3,
        n2=100,
        n3=2,
        p=0.5,
    )
    symbol = "EURUSD"
    strategy._ensure_symbol_state(symbol)
    strategy._saved_zones_by_symbol[symbol] = [{"top": 105.0, "bot": 95.0, "bar": 0}]

    entry_df = make_df(
        closes=[80.0, 82.0, 83.0, 106.0, 110.0, 111.0],
        highs=[80.5, 82.5, 83.5, 106.5, 110.5, 111.5],
        lows=[79.5, 81.5, 82.5, 105.5, 109.5, 110.5],
        symbol=symbol,
    )
    zone_df = make_df([100.0, 100.0, 100.0, 100.0, 100.0, 100.0], symbol=symbol)
    signals = strategy.generate_signals({"M1": entry_df, "M5": zone_df})
    assert signals == []


def test_breakout_long_emits_buy_with_tp_sl():
    strategy, broker, data = build_breakout_context()
    signals = strategy.generate_signals(data)
    assert len(signals) == 1
    sig = signals[0]
    assert sig.signal_type == SignalType.BUY
    assert sig.symbol == "EURUSD"
    assert sig.strategy_name == "PivotZoneTest"
    assert sig.timeframe == "M1"
    assert sig.size > 0
    assert sig.stop_loss == pytest.approx(99.0)
    assert sig.take_profit == pytest.approx(109.0)


def test_bracket_created_without_trailing_updates_stop_on_long():
    broker = FakeBrokerClient()
    strategy = PivotZoneTestStrategy(
        name="PivotZoneTest",
        broker_client=broker,
        tf_entry="M1",
        tf_zone="M5",
        tf_stop="M5",
    )
    symbol = "EURUSD"
    strategy._ensure_symbol_state(symbol)
    magic = strategy._get_magic_number()
    broker.positions.append(
        Position(
            symbol=symbol,
            volume=0.2,
            entry_price=1.5,
            stop_loss=1.0,
            take_profit=2.0,
            strategy_name=strategy.name,
            open_time=pd.Timestamp("2024-01-01"),
            magic_number=magic,
        )
    )
    state = strategy._trade_state[symbol]
    state.in_position = True
    state.direction = 1
    state.active_stop = 1.0
    state.active_tp = 2.0
    state.broken_zone = {"top": 1.1, "bot": 0.9}
    state.target_zone = {"top": 2.1, "bot": 1.9}
    # nuevo pivote disponible, pero sin trailing activo no debe mover el SL.
    strategy._stop_last_min[symbol] = 1.2
    strategy._stop_pivot_mins[symbol] = [1.0, 1.2]

    entry_df = make_df([1.5, 1.5, 1.5], symbol=symbol)
    zone_df = make_df([1.0, 1.0, 1.0], symbol=symbol)

    signals = strategy.generate_signals({"M1": entry_df, "M5": zone_df})
    assert signals == []
    assert len(broker.pending_orders) == 2
    sl_order = next(o for o in broker.pending_orders if "STOP" in o.order_type)
    tp_order = next(o for o in broker.pending_orders if "LIMIT" in o.order_type)
    assert sl_order.price == pytest.approx(1.0)
    assert tp_order.price == pytest.approx(2.0)
    assert state.stop_update_pending is False
    assert state.pending_stop_price is None


def test_breakout_long_descarta_si_no_hay_stop_confirmado_dentro_de_zona():
    strategy, _, data = build_breakout_context()
    symbol = "EURUSD"
    # pivotes confirmados fuera de la zona rota [99, 101]
    strategy._stop_pivot_mins[symbol] = [98.5, 102.2]

    signals = strategy.generate_signals(data)
    assert signals == []


def test_bracket_cancelled_when_position_missing_short():
    broker = FakeBrokerClient()
    strategy = PivotZoneTestStrategy(
        name="PivotZoneTest",
        broker_client=broker,
        tf_entry="M1",
        tf_zone="M5",
        tf_stop="M5",
    )
    symbol = "EURUSD"
    strategy._ensure_symbol_state(symbol)
    state = strategy._trade_state[symbol]
    state.in_position = True
    state.direction = -1
    state.active_stop = 1.5
    state.active_tp = 0.9
    state.tp_order_id = 10
    state.sl_order_id = 11
    # Ordenes pendientes simuladas existentes
    broker.pending_orders.extend(
        [
            SimpleNamespace(order_id=10, symbol=symbol, order_type="BUY_LIMIT"),
            SimpleNamespace(order_id=11, symbol=symbol, order_type="BUY_STOP"),
        ]
    )

    entry_df = make_df([1.5, 1.5, 1.5], symbol=symbol)
    zone_df = make_df([1.0, 1.0, 1.0], symbol=symbol)

    _ = strategy.generate_signals({"M1": entry_df, "M5": zone_df})
    assert broker.pending_orders == []
    state_after = strategy._trade_state[symbol]
    assert state_after.in_position is False


def test_determinism():
    strategy_a, broker_a, data_a = build_breakout_context()
    strategy_b, broker_b, data_b = build_breakout_context()

    signals_a = strategy_a.generate_signals(data_a)
    signals_b = strategy_b.generate_signals(data_b)

    assert signals_a == signals_b


def test_resolve_params_applies_symbol_overrides():
    broker = FakeBrokerClient()
    strategy = PivotZoneTestStrategy(
        name="PivotZoneTest",
        broker_client=broker,
        tf_entry="M1",
        tf_zone="M3",
        tf_stop="M1",
        n1=10,
        n2=80,
        n3=4,
        size_pct=0.04,
        p=0.45,
        symbol_params={
            "GBPUSD": {"n1": 21, "n2": 150, "n3": 6, "size_pct": 0.1, "p": 0.35}
        },
    )

    params_gbp = strategy._resolve_params("GBPUSD")
    assert params_gbp.n1 == 21
    assert params_gbp.n2 == 150
    assert params_gbp.n3 == 6
    assert params_gbp.size_pct == pytest.approx(0.1)
    assert params_gbp.p == pytest.approx(0.35)

    params_eur = strategy._resolve_params("EURUSD")
    assert params_eur.n1 == 10
    assert params_eur.n2 == 80
    assert params_eur.n3 == 4
    assert params_eur.size_pct == pytest.approx(0.04)
    assert params_eur.p == pytest.approx(0.45)


def test_symbol_size_override_changes_lot_size():
    base_strategy, _, base_data = build_breakout_context()
    base_signals = base_strategy.generate_signals(base_data)
    assert base_signals, "La estrategia base debe generar señal"
    base_size = base_signals[0].size

    override_strategy, _, override_data = build_breakout_context(
        symbol_params={"EURUSD": {"size_pct": 0.10}}
    )
    override_signals = override_strategy.generate_signals(override_data)
    assert override_signals, "La estrategia con override debe generar señal"

    assert base_size == pytest.approx(0.01, rel=1e-3)
    assert override_signals[0].size == pytest.approx(0.03, rel=1e-3)
    assert override_signals[0].size > base_size


def test_lot_size_applies_notional_cap_by_size_pct():
    broker = FakeBrokerClient(equity=100_000.0)
    broker.symbol_info = SimpleNamespace(
        trade_tick_value=1.0,
        trade_tick_size=0.0001,
        trade_contract_size=100_000.0,
        volume_min=0.01,
        volume_step=0.01,
        volume_max=100.0,
    )
    strategy = PivotZoneTestStrategy(
        name="PivotZoneTest",
        broker_client=broker,
        tf_entry="M1",
        tf_zone="M5",
        tf_stop="M5",
        size_pct=0.05,
    )

    lots = strategy._calc_lot_size_by_stop(
        symbol="EURUSD",
        entry_price=1.35894,
        stop_price=1.35767,
        size_pct=0.05,
    )

    assert lots == pytest.approx(0.03, rel=1e-3)


def test_mt5_real_mode_no_crea_brackets_opuestos_y_limpia_pendientes():
    broker = MetaTrader5Client()
    strategy = PivotZoneTestStrategy(
        name="PivotZoneTest",
        broker_client=broker,
        tf_entry="M1",
        tf_zone="M5",
        tf_stop="M5",
    )
    symbol = "EURUSD"
    magic = strategy._get_magic_number()
    broker.positions.append(
        Position(
            symbol=symbol,
            volume=0.2,
            entry_price=1.5,
            stop_loss=1.0,
            take_profit=2.0,
            strategy_name=strategy.name,
            open_time=pd.Timestamp("2024-01-01"),
            magic_number=magic,
        )
    )
    broker.pending_orders.extend(
        [
            SimpleNamespace(order_id=10, symbol=symbol, order_type="SELL_LIMIT", magic_number=magic),
            SimpleNamespace(order_id=11, symbol=symbol, order_type="SELL_STOP", magic_number=magic),
        ]
    )
    strategy._ensure_symbol_state(symbol)
    state = strategy._trade_state[symbol]
    state.in_position = True
    state.direction = 1
    state.active_stop = 1.0
    state.active_tp = 2.0
    state.broken_zone = {"top": 1.1, "bot": 0.9}
    state.target_zone = {"top": 2.1, "bot": 1.9}
    state.tp_order_id = 10
    state.sl_order_id = 11

    entry_df = make_df([1.5, 1.5, 1.5], symbol=symbol)
    zone_df = make_df([1.0, 1.0, 1.0], symbol=symbol)
    signals = strategy.generate_signals({"M1": entry_df, "M5": zone_df})

    assert signals == []
    assert broker.pending_orders == []
