"""Tests con mocks para MetaTrader5Client.

Esta suite de tests usa mocks para simular las respuestas de MT5 y validar
la lógica del cliente sin requerir una conexión real al terminal.

Los tests cubren:
- Conexión exitosa y fallida
- Descarga de datos OHLCV con diferentes escenarios
- Envío de órdenes BUY/SELL/CLOSE con validaciones
- Consulta de posiciones y trades
- Manejo de errores y reconexiones
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from bot_trading.domain.entities import OrderRequest, OrderResult, Position, TradeRecord
from bot_trading.infrastructure.mt5_client import (
    MT5ConnectionError,
    MT5DataError,
    MT5OrderError,
    MetaTrader5Client,
)


# =============================================================================
# FIXTURES Y HELPERS
# =============================================================================


@pytest.fixture
def mock_mt5():
    """Fixture que mockea el módulo MetaTrader5."""
    with patch('bot_trading.infrastructure.mt5_client.mt5') as mock:
        yield mock


@pytest.fixture
def client():
    """Fixture que proporciona un cliente MT5."""
    return MetaTrader5Client(max_retries=3, retry_delay=0.1)


# =============================================================================
# TESTS DE CONEXIÓN
# =============================================================================


def test_connect_exitosa_actualiza_estado(mock_mt5, client):
    """Una conexión exitosa debe actualizar connected=True."""
    # Configurar mock
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock(name="MT5 Terminal")
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    
    # Ejecutar
    client.connect()
    
    # Verificar
    assert client.connected is True
    mock_mt5.initialize.assert_called_once()


def test_connect_fallida_lanza_excepcion(mock_mt5, client):
    """Si MT5 falla al inicializar, debe lanzar MT5ConnectionError."""
    # Configurar mock para fallar
    mock_mt5.initialize.return_value = False
    mock_mt5.last_error.return_value = (1, "Terminal not found")
    
    # Verificar que lanza la excepción correcta
    with pytest.raises(MT5ConnectionError) as exc_info:
        client.connect()
    
    assert "No se pudo inicializar MetaTrader5" in str(exc_info.value)
    assert client.connected is False


def test_ensure_connected_reconecta_si_es_necesario(mock_mt5, client):
    """_ensure_connected debe reconectar si connected=False."""
    # Configurar mock
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock(name="MT5 Terminal")
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    
    # Cliente no conectado
    assert client.connected is False
    
    # Llamar método privado
    client._ensure_connected()
    
    # Verificar que se conectó
    assert client.connected is True
    mock_mt5.initialize.assert_called_once()


# =============================================================================
# TESTS DE DESCARGA DE DATOS OHLCV
# =============================================================================


def test_get_ohlcv_timeframe_invalido_lanza_error(mock_mt5, client):
    """Debe validar el timeframe y lanzar ValueError si es inválido."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    
    client.connect()
    
    start = datetime(2024, 1, 1, 0, 0)
    end = datetime(2024, 1, 1, 1, 0)
    
    with pytest.raises(ValueError) as exc_info:
        client.get_ohlcv("EURUSD", "INVALID_TF", start, end)
    
    assert "no es válido" in str(exc_info.value)


def test_get_ohlcv_fechas_invertidas_lanza_error(mock_mt5, client):
    """Debe validar que start < end."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    
    client.connect()
    
    start = datetime(2024, 1, 2, 0, 0)
    end = datetime(2024, 1, 1, 0, 0)  # end antes de start
    
    with pytest.raises(ValueError) as exc_info:
        client.get_ohlcv("EURUSD", "M1", start, end)
    
    assert "debe ser anterior" in str(exc_info.value)


def test_get_ohlcv_simbolo_invalido_lanza_error(mock_mt5, client):
    """Debe validar que el símbolo existe."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    mock_mt5.symbol_info.return_value = None  # Símbolo no existe
    mock_mt5.last_error.return_value = (4301, "Symbol not found")
    
    client.connect()
    
    start = datetime(2024, 1, 1, 0, 0)
    end = datetime(2024, 1, 1, 1, 0)
    
    with pytest.raises(MT5DataError) as exc_info:
        client.get_ohlcv("INVALID_SYMBOL", "M1", start, end)
    
    assert "no existe o no está disponible" in str(exc_info.value)


def test_get_ohlcv_retorna_dataframe_con_columnas_correctas(mock_mt5, client):
    """El DataFrame debe tener las columnas correctas."""
    # Configurar mocks
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    
    mock_symbol_info = Mock()
    mock_symbol_info.visible = True
    mock_mt5.symbol_info.return_value = mock_symbol_info
    
    # Simular datos OHLCV
    mock_rates = [
        {
            'time': 1704067200,  # 2024-01-01 00:00:00
            'open': 1.1000,
            'high': 1.1010,
            'low': 1.0990,
            'close': 1.1005,
            'tick_volume': 100
        },
        {
            'time': 1704067260,  # 2024-01-01 00:01:00
            'open': 1.1005,
            'high': 1.1015,
            'low': 1.1000,
            'close': 1.1010,
            'tick_volume': 120
        }
    ]
    mock_mt5.copy_rates_range.return_value = mock_rates
    
    client.connect()
    
    start = datetime(2024, 1, 1, 0, 0)
    end = datetime(2024, 1, 1, 1, 0)
    
    # Ejecutar
    df = client.get_ohlcv("EURUSD", "M1", start, end)
    
    # Verificar
    assert isinstance(df, pd.DataFrame)
    # datetime ahora es el índice, no una columna
    assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert len(df) == 2
    assert df.iloc[0]['open'] == 1.1000
    assert df.iloc[0]['volume'] == 100


def test_get_ohlcv_normaliza_rango_mixto_naive_aware_a_utc(mock_mt5, client):
    """Debe aceptar start naive y end aware sin fallar por timezone."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)

    mock_symbol_info = Mock()
    mock_symbol_info.visible = True
    mock_mt5.symbol_info.return_value = mock_symbol_info
    mock_mt5.copy_rates_range.return_value = []

    client.connect()

    start_naive = datetime(2024, 1, 1, 0, 0)
    end_aware = datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc)

    df = client.get_ohlcv("EURUSD", "M1", start_naive, end_aware)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"


def test_get_ohlcv_sin_datos_retorna_dataframe_vacio(mock_mt5, client):
    """Si no hay datos, debe retornar DataFrame vacío con columnas correctas."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    
    mock_symbol_info = Mock()
    mock_symbol_info.visible = True
    mock_mt5.symbol_info.return_value = mock_symbol_info
    
    # Sin datos
    mock_mt5.copy_rates_range.return_value = []
    
    client.connect()
    
    start = datetime(2050, 1, 1, 0, 0)
    end = datetime(2050, 1, 2, 0, 0)
    
    df = client.get_ohlcv("EURUSD", "M1", start, end)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    # datetime ahora es el índice, no una columna
    assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert isinstance(df.index, pd.DatetimeIndex)


# =============================================================================
# TESTS DE ENVÍO DE ÓRDENES
# =============================================================================


def test_send_market_order_buy_exitosa(mock_mt5, client):
    """Una orden BUY exitosa debe retornar OrderResult con success=True."""
    # Configurar mocks
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    
    mock_symbol_info = Mock()
    mock_symbol_info.visible = True
    mock_symbol_info.volume_min = 0.01
    mock_symbol_info.volume_max = 100.0
    mock_symbol_info.volume_step = 0.01
    mock_symbol_info.filling_mode = 2  # ORDER_FILLING_FOK
    mock_mt5.symbol_info.return_value = mock_symbol_info
    mock_mt5.ORDER_FILLING_FOK = 2
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_RETURN = 0
    
    mock_tick = Mock()
    mock_tick.ask = 1.1000
    mock_tick.bid = 1.0998
    mock_mt5.symbol_info_tick.return_value = mock_tick
    
    # Resultado de orden exitosa
    mock_result = Mock()
    mock_result.retcode = 10009  # TRADE_RETCODE_DONE
    mock_result.order = 12345
    mock_result.volume = 0.1
    mock_result.price = 1.1000
    mock_mt5.order_send.return_value = mock_result
    mock_mt5.TRADE_RETCODE_DONE = 10009
    
    client.connect()
    
    order_request = OrderRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        stop_loss=1.0950,
        take_profit=1.1100,
        magic_number=12345
    )
    
    # Ejecutar
    result = client.send_market_order(order_request)
    
    # Verificar
    assert result.success is True
    assert result.order_id == 12345
    assert result.error_message is None
    mock_mt5.order_send.assert_called_once()


def test_send_market_order_buy_ajusta_stops_invalidos(mock_mt5, client):
    """BUY con SL/TP demasiado cercanos debe ajustarse antes del envío a MT5."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)

    mock_symbol_info = Mock()
    mock_symbol_info.visible = True
    mock_symbol_info.volume_min = 0.01
    mock_symbol_info.volume_max = 100.0
    mock_symbol_info.volume_step = 0.01
    mock_symbol_info.filling_mode = 1
    mock_symbol_info.point = 0.00001
    mock_symbol_info.trade_tick_size = 0.00001
    mock_symbol_info.digits = 5
    mock_symbol_info.trade_stops_level = 20
    mock_symbol_info.trade_freeze_level = 0
    mock_mt5.symbol_info.return_value = mock_symbol_info
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.TRADE_RETCODE_DONE = 10009

    mock_tick = Mock()
    mock_tick.ask = 1.30020
    mock_tick.bid = 1.30000
    mock_mt5.symbol_info_tick.return_value = mock_tick

    mock_result = Mock()
    mock_result.retcode = 10009
    mock_result.order = 555
    mock_result.volume = 0.03
    mock_result.price = 1.30020
    mock_mt5.order_send.return_value = mock_result

    client.connect()

    order_request = OrderRequest(
        symbol="GBPUSD",
        volume=0.03,
        order_type="BUY",
        stop_loss=1.29995,  # inválido por cercanía al bid
        take_profit=1.30010,  # inválido por cercanía al bid
        magic_number=777,
    )

    result = client.send_market_order(order_request)

    assert result.success is True
    sent_request = mock_mt5.order_send.call_args[0][0]
    assert sent_request["sl"] == pytest.approx(1.29979)
    assert sent_request["tp"] == pytest.approx(1.30021)


def test_send_market_order_omite_sl_si_no_hay_nivel_valido(mock_mt5, client):
    """Si no existe SL positivo válido por distancia mínima, se omite en el request."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)

    mock_symbol_info = Mock()
    mock_symbol_info.visible = True
    mock_symbol_info.volume_min = 0.01
    mock_symbol_info.volume_max = 100.0
    mock_symbol_info.volume_step = 0.01
    mock_symbol_info.filling_mode = 1
    mock_symbol_info.point = 0.00001
    mock_symbol_info.trade_tick_size = 0.00001
    mock_symbol_info.digits = 5
    mock_symbol_info.trade_stops_level = 20
    mock_symbol_info.trade_freeze_level = 0
    mock_mt5.symbol_info.return_value = mock_symbol_info
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.TRADE_RETCODE_DONE = 10009

    mock_tick = Mock()
    mock_tick.ask = 0.00022
    mock_tick.bid = 0.00020
    mock_mt5.symbol_info_tick.return_value = mock_tick

    mock_result = Mock()
    mock_result.retcode = 10009
    mock_result.order = 556
    mock_result.volume = 0.01
    mock_result.price = 0.00022
    mock_mt5.order_send.return_value = mock_result

    client.connect()

    order_request = OrderRequest(
        symbol="GBPUSD",
        volume=0.01,
        order_type="BUY",
        stop_loss=0.00010,
        take_profit=0.00030,
        magic_number=778,
    )

    result = client.send_market_order(order_request)

    assert result.success is True
    sent_request = mock_mt5.order_send.call_args[0][0]
    assert "sl" not in sent_request
    assert sent_request["tp"] == pytest.approx(0.00041)
    assert order_request.stop_loss is None


def test_send_market_order_volumen_invalido_lanza_error(mock_mt5, client):
    """Debe validar que el volumen sea positivo."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    
    client.connect()
    
    order_request = OrderRequest(
        symbol="EURUSD",
        volume=0.0,  # Volumen inválido
        order_type="BUY",
        magic_number=12345
    )
    
    with pytest.raises(ValueError) as exc_info:
        client.send_market_order(order_request)
    
    assert "debe ser mayor que 0" in str(exc_info.value)


def test_send_market_order_tipo_invalido_lanza_error(mock_mt5, client):
    """Debe validar el tipo de orden."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    
    client.connect()
    
    order_request = OrderRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="INVALID_TYPE",
        magic_number=12345
    )
    
    with pytest.raises(ValueError) as exc_info:
        client.send_market_order(order_request)
    
    assert "no válido" in str(exc_info.value)


def test_send_market_order_rechazada_retorna_error(mock_mt5, client):
    """Una orden rechazada debe retornar OrderResult con success=False."""
    # Configurar mocks
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    
    mock_symbol_info = Mock()
    mock_symbol_info.visible = True
    mock_symbol_info.volume_min = 0.01
    mock_symbol_info.volume_max = 100.0
    mock_symbol_info.volume_step = 0.01
    mock_symbol_info.filling_mode = 2  # ORDER_FILLING_FOK
    mock_mt5.symbol_info.return_value = mock_symbol_info
    mock_mt5.ORDER_FILLING_FOK = 2
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_RETURN = 0
    
    mock_tick = Mock()
    mock_tick.ask = 1.1000
    mock_mt5.symbol_info_tick.return_value = mock_tick
    
    # Orden rechazada
    mock_result = Mock()
    mock_result.retcode = 10013  # Invalid stops
    mock_result.comment = "Invalid stops"
    mock_mt5.order_send.return_value = mock_result
    mock_mt5.TRADE_RETCODE_DONE = 10009
    
    client.connect()
    
    order_request = OrderRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        magic_number=12345
    )
    
    result = client.send_market_order(order_request)
    
    assert result.success is False
    assert "rechazada" in result.error_message.lower()


def test_create_pending_order_ajusta_precio_sell_stop_invalido(mock_mt5, client):
    """Si SELL_STOP queda demasiado cerca/por encima del bid, debe ajustarse antes de enviar."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)

    mock_symbol_info = Mock()
    mock_symbol_info.visible = True
    mock_symbol_info.volume_min = 0.01
    mock_symbol_info.volume_max = 100.0
    mock_symbol_info.volume_step = 0.01
    mock_symbol_info.filling_mode = 1  # FOK
    mock_symbol_info.point = 0.00001
    mock_symbol_info.trade_tick_size = 0.00001
    mock_symbol_info.digits = 5
    mock_symbol_info.trade_stops_level = 20
    mock_symbol_info.trade_freeze_level = 0
    mock_mt5.symbol_info.return_value = mock_symbol_info

    mock_tick = Mock()
    mock_tick.bid = 1.36196
    mock_tick.ask = 1.36198
    mock_mt5.symbol_info_tick.return_value = mock_tick

    mock_mt5.ORDER_TYPE_BUY_LIMIT = 1
    mock_mt5.ORDER_TYPE_SELL_LIMIT = 2
    mock_mt5.ORDER_TYPE_BUY_STOP = 3
    mock_mt5.ORDER_TYPE_SELL_STOP = 4
    mock_mt5.ORDER_TYPE_BUY_STOP_LIMIT = 5
    mock_mt5.ORDER_TYPE_SELL_STOP_LIMIT = 6
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.TRADE_ACTION_PENDING = 10
    mock_mt5.ORDER_TIME_GTC = 20
    mock_mt5.TRADE_RETCODE_DONE = 10009

    mock_result = Mock()
    mock_result.retcode = 10009
    mock_result.order = 777
    mock_mt5.order_send.return_value = mock_result

    client.connect()

    result = client.create_pending_order(
        symbol="GBPUSD",
        order_type="SELL_STOP",
        volume=0.03,
        price=1.36196,  # inválido en borde del bid
        magic_number=1234,
        comment="test",
    )

    assert result.success is True
    sent_request = mock_mt5.order_send.call_args[0][0]
    assert sent_request["price"] < mock_tick.bid


def test_modify_order_no_changes_se_considera_exito(mock_mt5, client):
    """RetCode 10025 (No changes) no debe registrarse como fallo lógico."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)

    mock_symbol_info = Mock()
    mock_symbol_info.visible = True
    mock_symbol_info.filling_mode = 1
    mock_symbol_info.point = 0.00001
    mock_symbol_info.trade_tick_size = 0.00001
    mock_symbol_info.digits = 5
    mock_symbol_info.trade_stops_level = 20
    mock_symbol_info.trade_freeze_level = 0
    mock_mt5.symbol_info.return_value = mock_symbol_info

    mock_tick = Mock()
    mock_tick.bid = 1.36190
    mock_tick.ask = 1.36192
    mock_mt5.symbol_info_tick.return_value = mock_tick

    mock_mt5.ORDER_TYPE_SELL_STOP = 4
    mock_mt5.TRADE_ACTION_MODIFY = 11
    mock_mt5.TRADE_RETCODE_DONE = 10009

    existing_order = Mock()
    existing_order.symbol = "GBPUSD"
    existing_order.type = 4
    existing_order.price_open = 1.36150
    mock_mt5.orders_get.return_value = [existing_order]

    mock_result = Mock()
    mock_result.retcode = 10025
    mock_result.comment = "No changes"
    mock_mt5.order_send.return_value = mock_result

    client.connect()

    result = client.modify_order(order_id=123, price=1.36150)
    assert result.success is True
    assert result.order_id == 123


# =============================================================================
# TESTS DE CONSULTA DE POSICIONES
# =============================================================================


def test_get_open_positions_retorna_lista_vacia_sin_posiciones(mock_mt5, client):
    """Sin posiciones, debe retornar lista vacía."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    mock_mt5.positions_get.return_value = []
    
    client.connect()
    
    positions = client.get_open_positions()
    
    assert positions == []


def test_get_open_positions_retorna_lista_con_posiciones(mock_mt5, client):
    """Con posiciones, debe retornar lista de objetos Position."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    
    # Simular posiciones
    mock_pos1 = Mock()
    mock_pos1.symbol = "EURUSD"
    mock_pos1.volume = 0.1
    mock_pos1.price_open = 1.1000
    mock_pos1.sl = 1.0950
    mock_pos1.tp = 1.1100
    mock_pos1.comment = "Test Strategy"
    mock_pos1.time = 1704067200
    mock_pos1.magic = 12345
    
    mock_pos2 = Mock()
    mock_pos2.symbol = "GBPUSD"
    mock_pos2.volume = 0.2
    mock_pos2.price_open = 1.2500
    mock_pos2.sl = 0  # Sin SL
    mock_pos2.tp = 0  # Sin TP
    mock_pos2.comment = "Another Strategy"
    mock_pos2.time = 1704067300
    mock_pos2.magic = 54321
    
    mock_mt5.positions_get.return_value = [mock_pos1, mock_pos2]
    
    client.connect()
    
    positions = client.get_open_positions()
    
    assert len(positions) == 2
    assert isinstance(positions[0], Position)
    assert positions[0].symbol == "EURUSD"
    assert positions[0].volume == 0.1
    assert positions[0].stop_loss == 1.0950
    assert positions[1].stop_loss is None  # SL = 0 se convierte a None


# =============================================================================
# TESTS DE CONSULTA DE TRADES CERRADOS
# =============================================================================


def test_get_closed_trades_retorna_lista_vacia_sin_trades(mock_mt5, client):
    """Sin trades, debe retornar lista vacía."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    mock_mt5.history_deals_get.return_value = []
    
    client.connect()
    
    trades = client.get_closed_trades()
    
    assert trades == []


def test_get_closed_trades_construye_trades_completos(mock_mt5, client):
    """Debe agrupar deals de entrada y salida en trades completos."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    mock_mt5.DEAL_ENTRY_IN = 0
    mock_mt5.DEAL_ENTRY_OUT = 1
    
    # Simular deals
    mock_deal_in = Mock()
    mock_deal_in.position_id = 1001
    mock_deal_in.entry = 0  # DEAL_ENTRY_IN
    mock_deal_in.symbol = "EURUSD"
    mock_deal_in.magic = 12345
    mock_deal_in.comment = "Test Strategy"
    mock_deal_in.time = 1704067200
    mock_deal_in.price = 1.1000
    mock_deal_in.volume = 0.1
    
    mock_deal_out = Mock()
    mock_deal_out.position_id = 1001
    mock_deal_out.entry = 1  # DEAL_ENTRY_OUT
    mock_deal_out.symbol = "EURUSD"
    mock_deal_out.magic = 12345
    mock_deal_out.time = 1704153600  # 1 día después
    mock_deal_out.price = 1.1050
    mock_deal_out.profit = 50.0
    
    mock_mt5.history_deals_get.return_value = [mock_deal_in, mock_deal_out]
    
    client.connect()
    
    trades = client.get_closed_trades()
    
    assert len(trades) == 1
    assert isinstance(trades[0], TradeRecord)
    assert trades[0].symbol == "EURUSD"
    assert trades[0].entry_price == 1.1000
    assert trades[0].exit_price == 1.1050
    assert trades[0].pnl == 50.0
    assert trades[0].size == 0.1


# =============================================================================
# TESTS DE CACHE DE SYMBOL INFO
# =============================================================================


def test_get_symbol_info_cachea_informacion(mock_mt5, client):
    """Debe cachear la información de símbolos para optimizar consultas."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    
    mock_symbol_info = Mock()
    mock_symbol_info.visible = True
    mock_symbol_info.spread = 10
    mock_symbol_info.volume_min = 0.01
    mock_mt5.symbol_info.return_value = mock_symbol_info
    
    client.connect()
    
    # Primera llamada
    info1 = client._get_symbol_info("EURUSD")
    
    # Segunda llamada (debe usar cache)
    info2 = client._get_symbol_info("EURUSD")
    
    # Verificar que solo se llamó una vez a MT5
    assert mock_mt5.symbol_info.call_count == 1
    assert info1 == info2


# =============================================================================
# TESTS DE DESTRUCTOR
# =============================================================================


def test_destructor_cierra_conexion(mock_mt5, client):
    """El destructor debe cerrar la conexión con MT5."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    
    client.connect()
    assert client.connected is True
    
    # Llamar destructor
    client.__del__()
    
    # Verificar que se cerró
    mock_mt5.shutdown.assert_called_once()
    assert client.connected is False

