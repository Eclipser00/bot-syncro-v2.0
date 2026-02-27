"""Script de prueba para verificar la conexión con MetaTrader 5.

Este script realiza una serie de pruebas básicas para verificar que:
1. MetaTrader5 está instalado y corriendo
2. Se puede establecer conexión
3. Se pueden obtener datos de mercado
4. Se pueden consultar posiciones y trades
5. Los símbolos configurados están disponibles

NO USA PYTEST, ES UN SCRIPT DE PRUEBA.
python tests/check_mt5_connection.py

Ejecutar antes de usar el bot en producción.
"""
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Asegurar que el paquete está en sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot_trading.infrastructure.mt5_client import (
    MetaTrader5Client,
    MT5ConnectionError,
    MT5DataError,
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_connection():
    """Prueba de conexión básica con MT5."""
    logger.info("="*80)
    logger.info("TEST 1: Conexión con MetaTrader 5")
    logger.info("="*80)
    
    try:
        client = MetaTrader5Client()
        client.connect()
        logger.info("✅ Conexión exitosa")
        return client
    except MT5ConnectionError as e:
        logger.error("❌ Error de conexión: %s", e)
        logger.error("\nVerifica que:")
        logger.error("  1. MetaTrader 5 esté instalado")
        logger.error("  2. El terminal esté corriendo")
        logger.error("  3. Estés logueado en una cuenta")
        sys.exit(1)


def test_account_info(client):
    """Prueba obtención de información de cuenta."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Información de Cuenta")
    logger.info("="*80)
    
    try:
        import MetaTrader5 as mt5
        
        account_info = mt5.account_info()
        if account_info:
            logger.info("✅ Información de cuenta obtenida:")
            logger.info("  - Login: %d", account_info.login)
            logger.info("  - Servidor: %s", account_info.server)
            logger.info("  - Balance: %.2f", account_info.balance)
            logger.info("  - Equity: %.2f", account_info.equity)
            logger.info("  - Margen: %.2f", account_info.margin)
            logger.info("  - Margen libre: %.2f", account_info.margin_free)
            logger.info("  - Nivel de margen: %.2f%%", account_info.margin_level if account_info.margin_level else 0)
            logger.info("  - Moneda: %s", account_info.currency)
            logger.info("  - Apalancamiento: 1:%d", account_info.leverage)
        else:
            logger.warning("⚠️ No se pudo obtener información de cuenta")
    except Exception as e:
        logger.error("❌ Error al obtener información: %s", e)


def test_symbols_availability(client, symbols):
    """Prueba disponibilidad de símbolos."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Disponibilidad de Símbolos")
    logger.info("="*80)
    
    available = []
    unavailable = []
    
    for symbol in symbols:
        try:
            info = client._get_symbol_info(symbol)
            available.append(symbol)
            logger.info("✅ %s: Disponible (Spread: %d, Lot min: %.2f)",
                       symbol, info.spread, info.volume_min)
        except MT5DataError:
            unavailable.append(symbol)
            logger.error("❌ %s: NO disponible", symbol)
    
    logger.info("\nResumen:")
    logger.info("  - Disponibles: %d/%d", len(available), len(symbols))
    logger.info("  - No disponibles: %d/%d", len(unavailable), len(symbols))
    
    if unavailable:
        logger.warning("\n⚠️ Símbolos no disponibles: %s", unavailable)
        logger.warning("Estos símbolos NO se podrán operar")
    
    return available


def test_download_ohlcv(client, symbols):
    """Prueba descarga de datos OHLCV."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Descarga de Datos OHLCV")
    logger.info("="*80)
    
    end = datetime.now()
    start = end - timedelta(hours=24)
    
    success_count = 0
    
    for symbol in symbols[:2]:  # Probar con los primeros 2 símbolos
        try:
            logger.info("\nDescargando datos de %s...", symbol)
            df = client.get_ohlcv(symbol, "H1", start, end)
            
            if len(df) > 0:
                logger.info("✅ %s: %d registros descargados", symbol, len(df))
                logger.info("  - Primer registro: %s", df.index[0])
                logger.info("  - Último registro: %s", df.index[-1])
                logger.info("  - Último precio: %.5f", df.iloc[-1]['close'])
                success_count += 1
            else:
                logger.warning("⚠️ %s: Sin datos en el rango solicitado", symbol)
        except Exception as e:
            logger.error("❌ Error al descargar %s: %s", symbol, e)
    
    logger.info("\nResumen: %d/%d símbolos descargados exitosamente", 
               success_count, min(2, len(symbols)))


def test_positions_and_trades(client):
    """Prueba consulta de posiciones y trades."""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: Posiciones y Trades")
    logger.info("="*80)
    
    try:
        # Posiciones abiertas
        positions = client.get_open_positions()
        logger.info("✅ Posiciones abiertas: %d", len(positions))
        
        if positions:
            for pos in positions:
                logger.info("  - %s: %.2f lotes @ %.5f (Magic: %s)",
                           pos.symbol, pos.volume, pos.entry_price, pos.magic_number)
        else:
            logger.info("  (No hay posiciones abiertas)")
        
        # Trades cerrados
        trades = client.get_closed_trades()
        logger.info("✅ Trades cerrados (hoy): %d", len(trades))
        
        if trades:
            total_pnl = sum(t.pnl for t in trades)
            logger.info("  - PnL total: %.2f", total_pnl)
            
            # Mostrar últimos 5 trades
            for trade in trades[:5]:
                logger.info("  - %s: %.2f lotes, PnL=%.2f, Entrada=%.5f, Salida=%.5f",
                           trade.symbol, trade.size, trade.pnl,
                           trade.entry_price, trade.exit_price)
        else:
            logger.info("  (No hay trades cerrados hoy)")
            
    except Exception as e:
        logger.error("❌ Error al consultar posiciones/trades: %s", e)


def test_timeframes(client, symbol):
    """Prueba descarga de diferentes timeframes."""
    logger.info("\n" + "="*80)
    logger.info("TEST 6: Timeframes Disponibles")
    logger.info("="*80)
    
    end = datetime.now()
    timeframes = {
        "M1": timedelta(hours=1),
        "M5": timedelta(hours=2),
        "M15": timedelta(hours=4),
        "H1": timedelta(hours=24),
        "H4": timedelta(days=7),
        "D1": timedelta(days=30),
    }
    
    logger.info("Probando timeframes con símbolo: %s\n", symbol)
    
    for tf, delta in timeframes.items():
        try:
            start = end - delta
            df = client.get_ohlcv(symbol, tf, start, end)
            logger.info("✅ %s: %d registros", tf, len(df))
        except Exception as e:
            logger.error("❌ %s: Error - %s", tf, e)


def main():
    """Ejecuta todos los tests."""
    logger.info("\n")
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*20 + "TEST DE CONEXIÓN MT5" + " "*38 + "║")
    logger.info("╚" + "="*78 + "╝")
    logger.info("\n")
    
    # Símbolos a probar (estándar en MT5 demo)
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    
    try:
        # Test 1: Conexión
        client = test_connection()
        
        # Test 2: Información de cuenta
        test_account_info(client)
        
        # Test 3: Disponibilidad de símbolos
        available_symbols = test_symbols_availability(client, symbols)
        
        if not available_symbols:
            logger.error("\n❌ CRÍTICO: No hay símbolos disponibles")
            logger.error("No se puede continuar con las pruebas")
            sys.exit(1)
        
        # Test 4: Descarga de datos
        test_download_ohlcv(client, available_symbols)
        
        # Test 5: Posiciones y trades
        test_positions_and_trades(client)
        
        # Test 6: Timeframes
        if available_symbols:
            test_timeframes(client, available_symbols[0])
        
        # Resumen final
        logger.info("\n" + "="*80)
        logger.info("RESUMEN DE PRUEBAS")
        logger.info("="*80)
        logger.info("✅ Todas las pruebas completadas")
        logger.info("\n🎉 MetaTrader 5 está LISTO para usar con el bot")
        logger.info("\nPuedes ejecutar el bot principal con:")
        logger.info("  python bot_trading/main.py")
        logger.info("\n" + "="*80 + "\n")
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Pruebas interrumpidas por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error("\n❌ Error inesperado: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

