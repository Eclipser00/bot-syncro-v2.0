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
    TIMEFRAME_MAP,
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
def client(tmp_path):
    """Fixture que proporciona un cliente MT5."""
    return MetaTrader5Client(
        max_retries=3,
        retry_delay=0.1,
        closed_trades_cursor_path=str(tmp_path / "closed_trades_cursor.json"),
    )


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
    
    msg = str(exc_info.value)
    assert ("no es válido" in msg) or ("no es vÃ¡lido" in msg)


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
    
    msg = str(exc_info.value)
    assert ("no existe o no está disponible" in msg) or ("no existe o no estÃ¡ disponible" in msg)


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


def test_get_ohlcv_m3_valida_disponibilidad_runtime(mock_mt5, client):
    """M3 debe funcionar si MT5 lo soporta, o fallar con error claro si no está disponible."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)

    mock_symbol_info = Mock()
    mock_symbol_info.visible = True
    mock_mt5.symbol_info.return_value = mock_symbol_info
    mock_mt5.copy_rates_range.return_value = []

    client.connect()

    start = datetime(2024, 1, 1, 0, 0)
    end = datetime(2024, 1, 1, 1, 0)

    if TIMEFRAME_MAP.get("M3") is None:
        with pytest.raises(ValueError, match="no est"):
            client.get_ohlcv("EURUSD", "M3", start, end)
    else:
        df = client.get_ohlcv("EURUSD", "M3", start, end)
        assert isinstance(df, pd.DataFrame)


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
    
    msg = str(exc_info.value)
    assert ("no válido" in msg) or ("no vÃ¡lido" in msg)


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


def test_get_closed_trades_usa_pnl_neto_realizado(mock_mt5, client):
    """El PnL del trade debe incluir profit, commission, swap y fee."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    mock_mt5.DEAL_ENTRY_IN = 0
    mock_mt5.DEAL_ENTRY_OUT = 1

    mock_deal_in = Mock()
    mock_deal_in.position_id = 1001
    mock_deal_in.ticket = 7001
    mock_deal_in.entry = 0
    mock_deal_in.symbol = "EURUSD"
    mock_deal_in.magic = 12345
    mock_deal_in.comment = "Test Strategy"
    mock_deal_in.time = 1704067200
    mock_deal_in.price = 1.1000
    mock_deal_in.volume = 0.1

    mock_deal_out = Mock()
    mock_deal_out.position_id = 1001
    mock_deal_out.ticket = 7002
    mock_deal_out.entry = 1
    mock_deal_out.symbol = "EURUSD"
    mock_deal_out.magic = 12345
    mock_deal_out.comment = "Test Strategy"
    mock_deal_out.time = 1704067260
    mock_deal_out.price = 1.1050
    mock_deal_out.volume = 0.1
    mock_deal_out.profit = 50.0
    mock_deal_out.commission = -2.0
    mock_deal_out.swap = -1.5
    mock_deal_out.fee = -0.5

    mock_mt5.history_deals_get.return_value = [mock_deal_in, mock_deal_out]

    client.connect()
    trades = client.get_closed_trades()

    assert len(trades) == 1
    assert trades[0].pnl == 46.0


def test_get_closed_trades_consulta_history_con_datetimes_utc_aware(mock_mt5, client):
    """history_deals_get debe recibir rango UTC-aware en get_closed_trades."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    mock_mt5.DEAL_ENTRY_IN = 0
    mock_mt5.DEAL_ENTRY_OUT = 1

    symbol_info = Mock()
    symbol_info.visible = True
    mock_mt5.symbol_info.return_value = symbol_info
    mock_mt5.symbol_info_tick.return_value = Mock(time=int(datetime(2026, 3, 5, 8, 0, tzinfo=timezone.utc).timestamp()))

    captured: dict[str, datetime] = {}

    def _history_deals_get(*args, **kwargs):
        if args:
            captured["from"] = args[0]
            captured["to"] = args[1]
        return []

    mock_mt5.history_deals_get.side_effect = _history_deals_get
    client.connect()

    _ = client.get_closed_trades()

    assert "from" in captured and "to" in captured
    assert captured["from"].tzinfo is not None
    assert captured["to"].tzinfo is not None
    assert captured["from"].utcoffset() == timezone.utc.utcoffset(captured["from"])
    assert captured["to"].utcoffset() == timezone.utc.utcoffset(captured["to"])


def test_get_closed_trades_reconstruye_trade_cross_day(mock_mt5, client):
    """IN en dia D y OUT en D+1 debe reconstruir trade completo."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    mock_mt5.DEAL_ENTRY_IN = 0
    mock_mt5.DEAL_ENTRY_OUT = 1

    symbol_info = Mock()
    symbol_info.visible = True
    mock_mt5.symbol_info.return_value = symbol_info
    mock_mt5.symbol_info_tick.return_value = Mock(time=int(datetime(2026, 3, 5, 8, 0, tzinfo=timezone.utc).timestamp()))

    mock_deal_in = Mock()
    mock_deal_in.position_id = 55560004296
    mock_deal_in.ticket = 101
    mock_deal_in.entry = 0
    mock_deal_in.symbol = "EURUSD"
    mock_deal_in.magic = 99001
    mock_deal_in.comment = "PivotZoneTest-M3"
    mock_deal_in.time = int(datetime(2026, 3, 4, 12, 9, 1, tzinfo=timezone.utc).timestamp())
    mock_deal_in.price = 1.08450
    mock_deal_in.volume = 0.20

    mock_deal_out = Mock()
    mock_deal_out.position_id = 55560004296
    mock_deal_out.ticket = 202
    mock_deal_out.entry = 1
    mock_deal_out.symbol = "EURUSD"
    mock_deal_out.magic = 99001
    mock_deal_out.comment = "PivotZoneTest-M3"
    mock_deal_out.time = int(datetime(2026, 3, 5, 6, 29, 37, tzinfo=timezone.utc).timestamp())
    mock_deal_out.price = 1.08200
    mock_deal_out.volume = 0.20
    mock_deal_out.profit = -50.0

    mock_mt5.history_deals_get.return_value = [mock_deal_in, mock_deal_out]
    client.connect()

    trades = client.get_closed_trades()

    assert len(trades) == 1
    trade = trades[0]
    assert trade.position_id == 55560004296
    assert trade.entry_deal_ticket == 101
    assert trade.exit_deal_ticket == 202
    assert trade.entry_time == datetime(2026, 3, 4, 12, 9, 1, tzinfo=timezone.utc)
    assert trade.exit_time == datetime(2026, 3, 5, 6, 29, 37, tzinfo=timezone.utc)


def test_get_closed_trades_snapshot_consulta_rango_aware_y_no_muta_cursor(mock_mt5, client, tmp_path):
    """Snapshot historico debe consultar UTC-aware y no persistir cursor."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    mock_mt5.DEAL_ENTRY_IN = 0
    mock_mt5.DEAL_ENTRY_OUT = 1

    captured: dict[str, datetime] = {}

    def _history_deals_get(*args, **kwargs):
        if args:
            captured["from"] = args[0]
            captured["to"] = args[1]
        return []

    mock_mt5.history_deals_get.side_effect = _history_deals_get
    client.connect()

    from_utc = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    to_utc = datetime(2026, 3, 5, 0, 0, tzinfo=timezone.utc)
    trades = client.get_closed_trades_snapshot(from_utc, to_utc)

    assert trades == []
    assert captured["from"].tzinfo is not None
    assert captured["to"].tzinfo is not None
    assert client._closed_trades_cursor_time_utc is None
    assert client._closed_trades_cursor_ticket == 0
    assert not (tmp_path / "closed_trades_cursor.json").exists()


def test_get_closed_trades_snapshot_reconstruye_cross_day_con_fallback(mock_mt5, client):
    """Snapshot debe reconstruir OUT sin IN en ventana usando fallback por position_id."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    mock_mt5.DEAL_ENTRY_IN = 0
    mock_mt5.DEAL_ENTRY_OUT = 1

    position_id = 987654321
    mock_deal_in = Mock()
    mock_deal_in.position_id = position_id
    mock_deal_in.ticket = 811
    mock_deal_in.entry = 0
    mock_deal_in.symbol = "EURUSD"
    mock_deal_in.magic = 99001
    mock_deal_in.comment = "PivotZoneTest-M3"
    mock_deal_in.time = int(datetime(2026, 3, 4, 12, 9, 1, tzinfo=timezone.utc).timestamp())
    mock_deal_in.price = 1.08450
    mock_deal_in.volume = 0.20

    mock_deal_out = Mock()
    mock_deal_out.position_id = position_id
    mock_deal_out.ticket = 812
    mock_deal_out.entry = 1
    mock_deal_out.symbol = "EURUSD"
    mock_deal_out.magic = 99001
    mock_deal_out.comment = "PivotZoneTest-M3"
    mock_deal_out.time = int(datetime(2026, 3, 5, 6, 29, 37, tzinfo=timezone.utc).timestamp())
    mock_deal_out.price = 1.08200
    mock_deal_out.volume = 0.20
    mock_deal_out.profit = -50.0

    def _history_deals_get(*args, **kwargs):
        if "position" in kwargs:
            return [mock_deal_in, mock_deal_out]
        return [mock_deal_out]

    mock_mt5.history_deals_get.side_effect = _history_deals_get
    client.connect()

    from_utc = datetime(2026, 3, 5, 0, 0, tzinfo=timezone.utc)
    to_utc = datetime(2026, 3, 6, 0, 0, tzinfo=timezone.utc)
    trades = client.get_closed_trades_snapshot(from_utc, to_utc)

    assert len(trades) == 1
    assert trades[0].entry_deal_ticket == 811
    assert trades[0].exit_deal_ticket == 812
    assert client._closed_trades_cursor_time_utc is None


def test_get_closed_trades_out_sin_in_en_ventana_recupera_entry_por_position_id(mock_mt5, client):
    """Si llega OUT sin IN en ventana, fallback por position_id debe recuperar IN."""
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    mock_mt5.DEAL_ENTRY_IN = 0
    mock_mt5.DEAL_ENTRY_OUT = 1

    symbol_info = Mock()
    symbol_info.visible = True
    mock_mt5.symbol_info.return_value = symbol_info
    mock_mt5.symbol_info_tick.return_value = Mock(time=int(datetime(2026, 3, 5, 8, 0, tzinfo=timezone.utc).timestamp()))

    position_id = 55560004296

    mock_deal_in = Mock()
    mock_deal_in.position_id = position_id
    mock_deal_in.ticket = 301
    mock_deal_in.entry = 0
    mock_deal_in.symbol = "EURUSD"
    mock_deal_in.magic = 99001
    mock_deal_in.comment = "PivotZoneTest-M3"
    mock_deal_in.time = int(datetime(2026, 3, 4, 12, 9, 1, tzinfo=timezone.utc).timestamp())
    mock_deal_in.price = 1.08450
    mock_deal_in.volume = 0.20

    mock_deal_out = Mock()
    mock_deal_out.position_id = position_id
    mock_deal_out.ticket = 302
    mock_deal_out.entry = 1
    mock_deal_out.symbol = "EURUSD"
    mock_deal_out.magic = 99001
    mock_deal_out.comment = "PivotZoneTest-M3"
    mock_deal_out.time = int(datetime(2026, 3, 5, 6, 29, 37, tzinfo=timezone.utc).timestamp())
    mock_deal_out.price = 1.08200
    mock_deal_out.volume = 0.20
    mock_deal_out.profit = -50.0

    def _history_deals_get(*args, **kwargs):
        if "position" in kwargs:
            return [mock_deal_in, mock_deal_out]
        return [mock_deal_out]

    mock_mt5.history_deals_get.side_effect = _history_deals_get
    client.connect()

    trades = client.get_closed_trades()

    assert len(trades) == 1
    assert trades[0].entry_deal_ticket == 301
    assert trades[0].exit_deal_ticket == 302


def test_get_closed_trades_cursor_persistente_evita_duplicados_tras_reinicio(mock_mt5, tmp_path):
    """Con cursor persistente, reiniciar cliente no debe duplicar cierres ya emitidos."""
    cursor_path = tmp_path / "closed_trades_cursor.json"

    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock()
    mock_mt5.account_info.return_value = Mock(login=12345, balance=10000.0)
    mock_mt5.DEAL_ENTRY_IN = 0
    mock_mt5.DEAL_ENTRY_OUT = 1

    symbol_info = Mock()
    symbol_info.visible = True
    mock_mt5.symbol_info.return_value = symbol_info
    mock_mt5.symbol_info_tick.return_value = Mock(time=int(datetime(2026, 3, 5, 8, 0, tzinfo=timezone.utc).timestamp()))

    mock_deal_in = Mock(
        position_id=777001,
        ticket=4001,
        entry=0,
        symbol="EURUSD",
        magic=99001,
        comment="PivotZoneTest-M3",
        time=int(datetime(2026, 3, 5, 1, 0, tzinfo=timezone.utc).timestamp()),
        price=1.0800,
        volume=0.10,
    )
    mock_deal_out = Mock(
        position_id=777001,
        ticket=4002,
        entry=1,
        symbol="EURUSD",
        magic=99001,
        comment="PivotZoneTest-M3",
        time=int(datetime(2026, 3, 5, 2, 0, tzinfo=timezone.utc).timestamp()),
        price=1.0810,
        volume=0.10,
        profit=10.0,
    )
    mock_mt5.history_deals_get.return_value = [mock_deal_in, mock_deal_out]

    client_1 = MetaTrader5Client(
        max_retries=3,
        retry_delay=0.1,
        closed_trades_cursor_path=str(cursor_path),
    )
    client_1.connect()
    trades_1 = client_1.get_closed_trades()

    client_2 = MetaTrader5Client(
        max_retries=3,
        retry_delay=0.1,
        closed_trades_cursor_path=str(cursor_path),
    )
    client_2.connect()
    trades_2 = client_2.get_closed_trades()

    assert len(trades_1) == 1
    assert trades_2 == []
    assert cursor_path.exists()


def test_strict_utc_mode_normaliza_naive_y_emite_warning(mock_mt5, client, caplog):
    """Modo UTC estricto debe normalizar datetime naive y dejar warning en logs."""
    naive_dt = datetime(2026, 3, 5, 6, 29, 37)

    with caplog.at_level("WARNING"):
        normalized = client._ensure_utc_datetime(naive_dt, caller="test_strict_mode")

    assert normalized.tzinfo is not None
    assert normalized.utcoffset() == timezone.utc.utcoffset(normalized)
    assert "Datetime naive detectado y normalizado a UTC" in caplog.text


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

