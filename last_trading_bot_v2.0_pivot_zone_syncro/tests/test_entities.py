"""Tests de entidades de dominio."""
from datetime import datetime

from bot_trading.domain.entities import Position, SymbolConfig


def test_symbol_config_attributes_definidos() -> None:
    """Valida que SymbolConfig expone los atributos esperados."""
    symbol = SymbolConfig(name="EURUSD", min_timeframe="M1")

    assert symbol.name == "EURUSD"
    assert symbol.min_timeframe == "M1"


def test_position_crea_instancia_completa() -> None:
    """Permite crear posiciones con todos los campos necesarios."""
    now = datetime.utcnow()
    position = Position(
        symbol="EURUSD",
        volume=0.1,
        entry_price=1.1,
        stop_loss=1.0,
        take_profit=1.2,
        strategy_name="demo",
        open_time=now,
    )

    assert position.symbol == "EURUSD"
    assert position.strategy_name == "demo"

