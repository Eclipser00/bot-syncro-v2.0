"""Tests de la capa de gestion de riesgo incremental."""
from datetime import datetime, timedelta, timezone

from bot_trading.application.risk_management import RiskManager
from bot_trading.domain.entities import RiskLimits, TradeRecord


def _build_trade(
    symbol: str,
    strategy: str,
    pnl: float,
    *,
    minutes_offset: int,
    exit_deal_ticket: int,
) -> TradeRecord:
    base = datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc)
    exit_time = base + timedelta(minutes=minutes_offset)
    return TradeRecord(
        symbol=symbol,
        strategy_name=strategy,
        entry_time=exit_time - timedelta(minutes=5),
        exit_time=exit_time,
        entry_price=1.0,
        exit_price=1.0,
        size=1.0,
        pnl=pnl,
        stop_loss=None,
        take_profit=None,
        exit_deal_ticket=exit_deal_ticket,
    )


def test_risk_manager_bloquea_bot_cuando_dd_actual_superado(tmp_path) -> None:
    manager = RiskManager(
        RiskLimits(
            dd_global=5.0,
            risk_state_path=str(tmp_path / "risk_state.json"),
        )
    )
    trades = [
        _build_trade("EURUSD", "strat", 1000.0, minutes_offset=0, exit_deal_ticket=101),
        _build_trade("EURUSD", "strat", -600.0, minutes_offset=1, exit_deal_ticket=102),
    ]

    manager.apply_closed_trades(trades, current_balance=10400.0)

    assert manager.global_state.anchor_balance == 11000.0
    assert manager.global_state.current_drawdown_percent == 600.0 / 11000.0 * 100.0
    assert manager.check_bot_risk_limits(10400.0) is False


def test_risk_manager_resetea_drawdown_global_con_nuevo_trade_ganador(tmp_path) -> None:
    manager = RiskManager(
        RiskLimits(
            dd_global=5.0,
            risk_state_path=str(tmp_path / "risk_state.json"),
        )
    )
    first_batch = [
        _build_trade("EURUSD", "strat", 1000.0, minutes_offset=0, exit_deal_ticket=101),
        _build_trade("EURUSD", "strat", -600.0, minutes_offset=1, exit_deal_ticket=102),
    ]
    manager.apply_closed_trades(first_batch, current_balance=10400.0)
    assert manager.check_bot_risk_limits(10400.0) is False

    second_batch = [_build_trade("EURUSD", "strat", 300.0, minutes_offset=2, exit_deal_ticket=103)]
    manager.apply_closed_trades(second_batch, current_balance=10700.0)

    assert manager.global_state.anchor_balance == 10700.0
    assert manager.global_state.current_drawdown_percent == 0.0
    assert manager.check_bot_risk_limits(10700.0) is True


def test_risk_manager_bloquea_activo_con_pnl_aislado_desde_ultimo_ganador(tmp_path) -> None:
    manager = RiskManager(
        RiskLimits(
            dd_por_activo={"EURUSD": 5.0},
            risk_state_path=str(tmp_path / "risk_state.json"),
        )
    )
    trades = [
        _build_trade("EURUSD", "strat", 500.0, minutes_offset=0, exit_deal_ticket=201),
        _build_trade("GBPUSD", "strat", -100.0, minutes_offset=1, exit_deal_ticket=202),
        _build_trade("EURUSD", "strat", -600.0, minutes_offset=2, exit_deal_ticket=203),
    ]

    manager.apply_closed_trades(trades, current_balance=9800.0)

    eurusd_state = manager.symbol_states["EURUSD"]
    assert eurusd_state.anchor_balance == 10500.0
    assert eurusd_state.isolated_pnl_since_anchor == -600.0
    assert eurusd_state.current_drawdown_percent == 600.0 / 10500.0 * 100.0
    assert manager.check_symbol_risk_limits("EURUSD") is False
    assert manager.check_symbol_risk_limits("GBPUSD") is True


def test_risk_manager_bloquea_estrategia_con_pnl_aislado_desde_ultimo_ganador(tmp_path) -> None:
    manager = RiskManager(
        RiskLimits(
            dd_por_estrategia={"trend": 5.0},
            risk_state_path=str(tmp_path / "risk_state.json"),
        )
    )
    trades = [
        _build_trade("EURUSD", "trend", 1000.0, minutes_offset=0, exit_deal_ticket=301),
        _build_trade("GBPUSD", "other", -200.0, minutes_offset=1, exit_deal_ticket=302),
        _build_trade("EURUSD", "trend", -600.0, minutes_offset=2, exit_deal_ticket=303),
    ]

    manager.apply_closed_trades(trades, current_balance=10200.0)

    trend_state = manager.strategy_states["trend"]
    assert trend_state.anchor_balance == 11000.0
    assert trend_state.isolated_pnl_since_anchor == -600.0
    assert trend_state.current_drawdown_percent == 600.0 / 11000.0 * 100.0
    assert manager.check_strategy_risk_limits("trend") is False
    assert manager.check_strategy_risk_limits("other") is True


def test_risk_manager_batch_multiple_ganadores_toma_ultimo_anchor(tmp_path) -> None:
    manager = RiskManager(
        RiskLimits(
            dd_global=10.0,
            risk_state_path=str(tmp_path / "risk_state.json"),
        )
    )
    trades = [
        _build_trade("EURUSD", "strat", 100.0, minutes_offset=0, exit_deal_ticket=401),
        _build_trade("EURUSD", "strat", -50.0, minutes_offset=1, exit_deal_ticket=402),
        _build_trade("EURUSD", "strat", 200.0, minutes_offset=2, exit_deal_ticket=403),
    ]

    manager.apply_closed_trades(trades, current_balance=10250.0)

    assert manager.global_state.anchor_balance == 10250.0
    assert manager.global_state.anchor_exit_deal_ticket == 403
    assert manager.global_state.current_drawdown_percent == 0.0


def test_risk_manager_persiste_y_recupera_estado(tmp_path) -> None:
    state_path = tmp_path / "risk_state.json"
    manager = RiskManager(RiskLimits(dd_global=5.0, risk_state_path=str(state_path)))
    trades = [
        _build_trade("EURUSD", "strat", 1000.0, minutes_offset=0, exit_deal_ticket=501),
        _build_trade("EURUSD", "strat", -600.0, minutes_offset=1, exit_deal_ticket=502),
    ]
    manager.apply_closed_trades(trades, current_balance=10400.0)

    reloaded = RiskManager(RiskLimits(dd_global=5.0, risk_state_path=str(state_path)))

    assert reloaded.global_state.anchor_balance == 11000.0
    assert reloaded.global_state.anchor_exit_deal_ticket == 501
    assert reloaded.global_state.current_drawdown_percent == 600.0 / 11000.0 * 100.0


def test_risk_manager_scope_sin_anchor_permanece_en_cero(tmp_path) -> None:
    manager = RiskManager(
        RiskLimits(
            dd_global=5.0,
            dd_por_activo={"EURUSD": 5.0},
            dd_por_estrategia={"trend": 5.0},
            risk_state_path=str(tmp_path / "risk_state.json"),
        )
    )
    trades = [_build_trade("EURUSD", "trend", -600.0, minutes_offset=0, exit_deal_ticket=601)]

    manager.apply_closed_trades(trades, current_balance=9400.0)

    assert manager.global_state.anchor_balance is None
    assert manager.global_state.current_drawdown_percent == 0.0
    assert manager.symbol_states["EURUSD"].current_drawdown_percent == 0.0
    assert manager.strategy_states["trend"].current_drawdown_percent == 0.0
    assert manager.check_bot_risk_limits(9400.0) is True
