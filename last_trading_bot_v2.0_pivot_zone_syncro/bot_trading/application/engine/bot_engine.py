"""Motor principal del bot de trading."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

from bot_trading.application.engine.order_executor import OrderExecutor
from bot_trading.application.engine.signals import SignalType
from bot_trading.application.risk_management import RiskManager
from bot_trading.application.strategy_registry import StrategyRegistry
from bot_trading.application.strategies.base import Strategy
from bot_trading.domain.entities import OrderRequest, SymbolConfig, TradeRecord
from bot_trading.infrastructure.data_fetcher import MarketDataService, DevelopmentCsvDataProvider

logger = logging.getLogger(__name__)


@dataclass
class TradingBot:
    """Coordina la ejecución completa del bot."""

    broker_client: object
    market_data_service: MarketDataService
    risk_manager: RiskManager
    order_executor: OrderExecutor
    strategies: list[Strategy]
    symbols: list[SymbolConfig]
    trade_history: list[TradeRecord] = field(default_factory=list)
    strategy_registry: StrategyRegistry = field(default_factory=StrategyRegistry)
    clock_offset: timedelta = field(default=timedelta(0))
    market_data_callback: Callable[[SymbolConfig, dict, datetime], None] | None = None
    _exhausted_symbols: set[str] = field(default_factory=set, init=False)
    _emitted_close_trade_keys: set[tuple] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        """Inicializa el registro de estrategias después de la construcción."""
        # Registrar todas las estrategias al iniciar
        for strategy in self.strategies:
            self.strategy_registry.register_strategy(strategy.name)

    def _now(self) -> datetime:
        """Devuelve la hora actual ajustada por el desfase del broker."""
        return datetime.now(timezone.utc) + self.clock_offset

    def run_once(self, now: datetime | None = None) -> None:
        """Ejecuta un ciclo completo del bot una sola vez."""
        current_time = now or self._now()
        logger.info("Iniciando ciclo del bot en %s", current_time)
        
        # Sincronizar estado de órdenes abiertas con el broker
        self.order_executor.sync_state()
        
        # Actualizar trade_history con trades cerrados del broker
        self._update_trade_history()
        
        if not self.risk_manager.check_bot_risk_limits(self.trade_history):
            logger.warning("Bot bloqueado por límites globales de riesgo")
            return

        for symbol in self.symbols:
            if symbol.name in self._exhausted_symbols:
                continue
            if not self.risk_manager.check_symbol_risk_limits(symbol.name, self.trade_history):
                logger.info("Símbolo %s bloqueado por riesgo", symbol.name)
                continue

            # Calcular timeframes requeridos SOLO para estrategias que operan este símbolo
            required_timeframes = set()
            for strategy in self.strategies:
                # Verificar si la estrategia puede operar este símbolo
                # Si allowed_symbols no existe o es None, la estrategia opera todos los símbolos
                can_operate = True
                if hasattr(strategy, 'allowed_symbols') and strategy.allowed_symbols is not None:
                    can_operate = symbol.name in strategy.allowed_symbols
                
                if can_operate:
                    required_timeframes.update(strategy.timeframes)
            
            # Si no hay timeframes requeridos para este símbolo, saltar
            if not required_timeframes:
                logger.debug("No hay estrategias que operen %s, saltando", symbol.name)
                continue
            
            # Filtrar timeframes incompatibles con el min_timeframe del símbolo
            # Solo mantener timeframes >= min_timeframe
            compatible_timeframes = set()
            tf_order = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
            try:
                min_tf_idx = tf_order.index(symbol.min_timeframe)
                for tf in required_timeframes:
                    if tf in tf_order:
                        tf_idx = tf_order.index(tf)
                        if tf_idx >= min_tf_idx:
                            compatible_timeframes.add(tf)
                    else:
                        # Si no está en la lista, incluir por seguridad
                        compatible_timeframes.add(tf)
            except ValueError:
                # Si min_timeframe no está en la lista, usar todos
                compatible_timeframes = required_timeframes
            
            required_timeframes = compatible_timeframes
            
            if not required_timeframes:
                logger.warning(
                    "No hay timeframes compatibles para %s (min_timeframe=%s). "
                    "Las estrategias requieren timeframes incompatibles.",
                    symbol.name, symbol.min_timeframe
                )
                continue
            
            try:
                data_by_timeframe = self.market_data_service.get_data(
                    symbol=symbol,
                    target_timeframes=required_timeframes,
                    now=current_time,
                )
                if self.market_data_callback is not None:
                    try:
                        self.market_data_callback(symbol, data_by_timeframe, current_time)
                    except Exception as callback_exc:
                        logger.debug(
                            "market_data_callback fallo para %s: %s",
                            symbol.name,
                            callback_exc,
                        )

                # Simular fills de SL/TP solo en modo desarrollo (CSV streaming)
                data_provider = getattr(self.market_data_service, "data_provider", None)
                if isinstance(data_provider, DevelopmentCsvDataProvider) and hasattr(self.broker_client, "process_price_tick"):
                    try:
                        df_price = data_by_timeframe.get(symbol.min_timeframe)
                        if df_price is None or df_price.empty:
                            df_price = next(iter(data_by_timeframe.values()))
                        if df_price is not None and not df_price.empty:
                            last_close = float(df_price.iloc[-1]["close"])
                            last_open = float(df_price.iloc[-1].get("open", last_close))
                            last_high = float(df_price.iloc[-1].get("high", last_close))
                            last_low = float(df_price.iloc[-1].get("low", last_close))
                            bar_idx = len(df_price) - 1
                            ts_event = df_price.index[-1].isoformat() if df_price.index is not None else current_time.isoformat()
                            tf_label = symbol.min_timeframe
                            self.broker_client.process_price_tick(symbol.name, last_close, last_high, last_low, last_open)
                            if hasattr(self.order_executor, "flush_pending_fills"):
                                try:
                                    self.order_executor.flush_pending_fills(now=current_time)
                                except Exception as e:
                                    logger.debug("flush_pending_fills fallo para %s: %s", symbol.name, e)
                            if hasattr(self.broker_client, "consume_closed_position_events"):
                                try:
                                    closed_events = self.broker_client.consume_closed_position_events()
                                except Exception:
                                    closed_events = []
                                should_sync_positions = False
                                for ev in closed_events or []:
                                    try:
                                        direction = int(ev.get("direction", 1))
                                        side = str(ev.get("fill_side") or ("short" if direction > 0 else "long"))
                                        size = abs(float(ev.get("volume", 0.0)))
                                        use_close_price = os.getenv("PARITY_CLOSE_PRICE_ON_CLOSE", "0") == "1"
                                        price = float(last_close) if use_close_price else float(ev.get("exit_price", last_close))
                                        symbol_name = str(ev.get("symbol") or symbol.name)
                                        magic_number = ev.get("magic_number")
                                        order_id = str(magic_number or "")
                                        reason = str(ev.get("reason") or "close")
                                        position_side = str(ev.get("position_side") or "flat")
                                        if position_side not in {"long", "short", "flat"}:
                                            position_side = "flat"
                                        position_size = abs(float(ev.get("position_size", 0.0)))
                                        if position_side == "flat":
                                            position_size = 0.0
                                        else:
                                            position_size = max(position_size, size)
                                        # order_fill sintético por SL/TP (paridad con backtest)
                                        self.order_executor._emit_event({
                                            "event_type": "order_fill",
                                            "ts_event": ts_event,
                                            "bar_index": bar_idx,
                                            "symbol": symbol_name,
                                            "timeframe": tf_label,
                                            "side": side,
                                            "price_ref": "close",
                                            "price": price,
                                            "size": size,
                                            "commission": 0.0,
                                            "slippage": 0.0,
                                            "reason": f"close_{reason}",
                                            "order_id": order_id,
                                        })
                                        # position tras el fill sintetico (flat o inversion)
                                        self.order_executor._emit_event({
                                            "event_type": "position",
                                            "ts_event": ts_event,
                                            "bar_index": bar_idx,
                                            "symbol": symbol_name,
                                            "timeframe": tf_label,
                                            "side": position_side,
                                            "price_ref": "close",
                                            "price": price,
                                            "size": position_size,
                                            "commission": 0.0,
                                            "slippage": 0.0,
                                            "reason": "position_update",
                                            "order_id": order_id,
                                        })
                                        try:
                                            if position_side == "flat":
                                                self.order_executor._remove_position(
                                                    symbol_name,
                                                    int(magic_number) if magic_number is not None else None,
                                                    emit_event=False,
                                                )
                                            else:
                                                should_sync_positions = True
                                        except Exception:
                                            pass
                                    except Exception:
                                        continue
                                if should_sync_positions:
                                    try:
                                        self.order_executor.sync_state()
                                    except Exception as e:
                                        logger.debug("sync_state tras cierres sinteticos fallo para %s: %s", symbol.name, e)
                    except Exception as e:
                        logger.debug("process_price_tick fallo para %s: %s", symbol.name, e)
            except StopIteration as e:
                logger.info("Datos de desarrollo agotados para %s: %s", symbol.name, e)
                self._exhausted_symbols.add(symbol.name)
                if len(self._exhausted_symbols) >= len(self.symbols):
                    raise
                continue
            except Exception as e:
                logger.error("Error obteniendo datos para %s: %s", symbol.name, e)
                continue

            for strategy in self.strategies:
                if not self.risk_manager.check_strategy_risk_limits(
                    strategy.name, self.trade_history
                ):
                    logger.info(
                        "Estrategia %s bloqueada por riesgo", strategy.name
                    )
                    continue

                signals = strategy.generate_signals(data_by_timeframe)
                logger.debug(
                    "Estrategia %s generó %d señales para %s",
                    strategy.name,
                    len(signals),
                    symbol.name,
                )
                
                # Obtener Magic Number de la estrategia (debe estar registrada)
                magic_number = self.strategy_registry.get_magic_number(strategy.name)
                if magic_number is None:
                    logger.error(
                        "Estrategia %s no tiene Magic Number asignado. Registrándola ahora.",
                        strategy.name
                    )
                    magic_number = self.strategy_registry.register_strategy(strategy.name)
                
                for signal in signals:
                    if signal.signal_type in {SignalType.BUY, SignalType.SELL}:
                        # Verificar si ya existe una posición abierta
                        if self.order_executor.has_open_position(
                            signal.symbol, 
                            strategy.name,
                            magic_number
                        ):
                            logger.debug(
                                "Orden %s ignorada: ya existe posición abierta para %s con estrategia %s",
                                signal.signal_type.value,
                                signal.symbol,
                                strategy.name,
                            )
                            continue
                        
                        # Validar límites de margen antes de crear la orden
                        try:
                            account_info = self.broker_client.get_account_info()
                            
                            # Calcular margen requerido si el broker lo soporta
                            required_margin = None
                            if hasattr(self.broker_client, '_calculate_required_margin'):
                                # Obtener precio actual para calcular margen
                                try:
                                    import MetaTrader5 as mt5
                                    tick = mt5.symbol_info_tick(signal.symbol)
                                    if tick:
                                        order_type = (
                                            mt5.ORDER_TYPE_BUY if signal.signal_type == SignalType.BUY
                                            else mt5.ORDER_TYPE_SELL
                                        )
                                        price = tick.ask if signal.signal_type == SignalType.BUY else tick.bid
                                        required_margin = self.broker_client._calculate_required_margin(
                                            signal.symbol, order_type, signal.size, price
                                        )
                                except Exception as e:
                                    logger.debug("No se pudo calcular margen requerido: %s", e)
                            
                            # Validar límites de margen
                            if not self.risk_manager.check_margin_limits(account_info, required_margin):
                                logger.warning(
                                    "Orden %s bloqueada por límites de margen para %s",
                                    signal.signal_type.value,
                                    signal.symbol,
                                )
                                continue
                        except AttributeError:
                            # El broker no implementa get_account_info, continuar sin validación
                            logger.debug("Broker no implementa get_account_info, omitiendo validación de margen")
                        except Exception as e:
                            logger.error("Error al validar límites de margen: %s", e)
                            # Por seguridad, bloquear la orden si hay error
                            continue
                        
                        tf_df = data_by_timeframe.get(signal.timeframe)
                        bar_index = len(tf_df) - 1 if tf_df is not None else -1
                        ts_event = None
                        if tf_df is not None and len(tf_df):
                            try:
                                ts_event = tf_df.index[-1].isoformat()
                            except Exception:
                                ts_event = None
                        timeframe_value = signal.timeframe
                        order_request = OrderRequest(
                            symbol=signal.symbol,
                            volume=signal.size,
                            order_type=signal.signal_type.value,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            comment=f"{signal.strategy_name}-{signal.timeframe}",
                            magic_number=magic_number,
                            bar_index=bar_index,
                            timeframe=timeframe_value,
                            price_ref='close',
                            ts_event=ts_event,
                        )
                        result = self.order_executor.execute_order(order_request)
                        if result.success:
                            logger.info("Orden ejecutada exitosamente: %s", result.order_id)
                    elif signal.signal_type == SignalType.CLOSE:
                        # Verificar existencia antes de intentar cerrar
                        if not self.order_executor.has_open_position(
                            signal.symbol, 
                            strategy.name,
                            magic_number
                        ):
                            logger.debug(
                                "Señal CLOSE ignorada: no existe posición abierta para %s", 
                                signal.symbol
                            )
                            continue

                        order_request = OrderRequest(
                            symbol=signal.symbol,
                            volume=signal.size,
                            order_type=signal.signal_type.value,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            comment=f"{signal.strategy_name}-{signal.timeframe}",
                            magic_number=magic_number,
                        )
                        self.order_executor.execute_order(order_request)
                    else:
                        logger.debug(
                            "Señal %s ignorada por ser tipo %s",
                            signal,
                            signal.signal_type,
                        )
    def run_forever(self, sleep_seconds: int = 60) -> None:
        """Ejecuta el bot en bucle infinito con pausas."""
        import time

        while True:
            self.run_once()
            time.sleep(sleep_seconds)

    def run_synchronized(
        self,
        timeframe_minutes: int = 1,
        wait_after_close: int = 5,
        skip_sleep_when_simulated: bool = False,
    ) -> None:
        """Ejecuta el bot sincronizado con el cierre de velas.
        
        Espera hasta que cierre la vela del timeframe especificado, luego espera
        wait_after_close segundos adicionales antes de ejecutar el ciclo.
        
        Esta función es ideal para operar en múltiples timeframes ya que sincroniza
        con el cierre de velas del timeframe base (típicamente M1).
        
        Args:
            timeframe_minutes: Duración de la vela en minutos (1=M1, 5=M5, 15=M15, 60=H1, etc.).
            wait_after_close: Segundos a esperar después del cierre de la vela.
            skip_sleep_when_simulated: si es True y hay reloj simulado, avanza por timestamps del CSV sin dormir.
        
        Ejemplo para M1:
            - Vela cierra a las 17:21:00
            - Espera hasta 17:21:05 (5 seg después)
            - Ejecuta run_once()
            - Espera hasta 17:22:05 (próxima vela + 5 seg)
            - Repite
        
        Ejemplo para M5:
            - Vela cierra a las 17:20:00, 17:25:00, 17:30:00...
            - Espera 5 seg después de cada cierre
            - Ejecuta run_once()
        """
        import time
        
        logger.info("="*80)
        logger.info("Iniciando bucle sincronizado con velas de %d minutos", timeframe_minutes)
        logger.info("Esperando %d segundos después del cierre de cada vela", wait_after_close)
        logger.info("Presiona Ctrl+C para detener el bot")
        logger.info("="*80)

        data_provider = getattr(self.market_data_service, "data_provider", None)
        get_simulated_now = getattr(data_provider, "get_simulated_now", None) if data_provider else None
        simulated_now = None
        has_simulated_clock = False
        if callable(get_simulated_now):
            try:
                simulated_now = get_simulated_now(self.symbols)
                has_simulated_clock = simulated_now is not None
            except Exception as exc:
                if skip_sleep_when_simulated:
                    logger.error(
                        "skip_sleep_when_simulated=True pero no se pudo obtener reloj simulado desde el proveedor: %s",
                        exc,
                    )
                    raise
                logger.debug("No se pudo detectar reloj simulado: %s", exc)

        if skip_sleep_when_simulated and callable(get_simulated_now) and not has_simulated_clock:
            raise RuntimeError(
                "skip_sleep_when_simulated=True pero el proveedor no devuelve reloj simulado; revisa CSV/entorno."
            )

        if has_simulated_clock:
            if skip_sleep_when_simulated:
                logger.info(
                    "Modo desarrollo detectado: usando timestamps del CSV (sin esperar al reloj real)."
                )
            else:
                logger.info(
                    "Modo desarrollo detectado pero skip_sleep_when_simulated=False; se usara reloj real con sleeps normales."
                )

        max_cycles_env = None
        try:
            max_cycles_env = int(os.environ.get("PARITY_MAX_CYCLES", "0"))
        except Exception:
            max_cycles_env = 0
        max_cycles = max_cycles_env if max_cycles_env and max_cycles_env > 0 else None
        cycles = 0

        if has_simulated_clock and skip_sleep_when_simulated:
            while True:
                try:
                    if max_cycles is not None and cycles >= max_cycles:
                        logger.info("PARITY_MAX_CYCLES alcanzado (%d). Bucle detenido", max_cycles)
                        break
                    simulated_now = get_simulated_now(self.symbols)
                    if simulated_now is None:
                        logger.warning("Reloj simulado no disponible; deteniendo bucle")
                        break

                    execution_time = simulated_now + timedelta(seconds=wait_after_close)
                    logger.info("-" * 80)
                    logger.info(
                        "Ejecutando ciclo de trading en %s (simulado)",
                        execution_time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    logger.info("-" * 80)
                    self.run_once(now=execution_time)
                    cycles += 1
                except StopIteration:
                    logger.info("Fin de datos de desarrollo (CSV). Bucle detenido")
                    if hasattr(self.broker_client, "flush_pending_closes"):
                        try:
                            self.broker_client.flush_pending_closes(now=execution_time)
                        except Exception as e:
                            logger.debug("flush_pending_closes fallo: %s", e)
                    break
                except KeyboardInterrupt:
                    logger.info("\nBucle interrumpido por el usuario")
                    break
                except Exception as e:
                    logger.error("Error en bucle desarrollo: %s", e, exc_info=True)
                    break
            return

        while True:
            try:
                # Calcular cuánto falta para el próximo cierre de vela
                now = self._now()

                # Calcular el próximo cierre según el timeframe
                if timeframe_minutes == 1:
                    # Para M1: próximo minuto
                    next_close = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
                elif timeframe_minutes == 5:
                    # Para M5: próximo múltiplo de 5 minutos
                    minutes_to_add = 5 - (now.minute % 5)
                    if minutes_to_add == 5:
                        minutes_to_add = 0
                    next_close = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
                elif timeframe_minutes == 15:
                    # Para M15: próximo múltiplo de 15 minutos
                    minutes_to_add = 15 - (now.minute % 15)
                    if minutes_to_add == 15:
                        minutes_to_add = 0
                    next_close = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
                elif timeframe_minutes == 30:
                    # Para M30: próximo múltiplo de 30 minutos
                    minutes_to_add = 30 - (now.minute % 30)
                    if minutes_to_add == 30:
                        minutes_to_add = 0
                    next_close = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
                elif timeframe_minutes == 60:
                    # Para H1: próxima hora
                    next_close = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                elif timeframe_minutes == 240:
                    # Para H4: próxima hora múltiplo de 4
                    hours_to_add = 4 - (now.hour % 4)
                    if hours_to_add == 4:
                        hours_to_add = 0
                    next_close = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=hours_to_add)
                else:
                    # Genérico: próximo múltiplo del timeframe
                    minutes_to_add = timeframe_minutes - (now.minute % timeframe_minutes)
                    if minutes_to_add == timeframe_minutes:
                        minutes_to_add = 0
                    next_close = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
                
                # Añadir tiempo de espera después del cierre
                execution_time = next_close + timedelta(seconds=wait_after_close)
                
                # Calcular segundos a esperar
                wait_seconds = (execution_time - now).total_seconds()
                
                if wait_seconds > 0:
                    logger.info(
                        "Esperando %.1f segundos hasta %s (cierre de vela + %d seg)",
                        wait_seconds, execution_time.strftime("%H:%M:%S"), wait_after_close
                    )
                    time.sleep(wait_seconds)
                elif wait_seconds < -1:
                    # Si ya pasó el tiempo, calcular el siguiente
                    logger.debug("Tiempo de ejecución ya pasó, calculando siguiente vela...")
                    continue

                # Ejecutar ciclo del bot
                logger.info("-"*80)
                logger.info("Ejecutando ciclo de trading en %s", self._now().strftime("%Y-%m-%d %H:%M:%S"))
                logger.info("-"*80)
                self.run_once()
                
            except KeyboardInterrupt:
                logger.info("\n⚠️ Bucle interrumpido por el usuario")
                break
            except Exception as e:
                logger.error("❌ Error en bucle sincronizado: %s", e, exc_info=True)
                # Esperar un poco antes de reintentar para evitar loops infinitos de errores
                logger.info("Esperando 10 segundos antes de reintentar...")
                time.sleep(10)

    def _update_trade_history(self) -> None:
        """Actualiza el historial de trades con los cerrados del broker.
        
        Usa una tupla con entry_time, exit_time, symbol y strategy_name para identificar
        trades únicos de forma más robusta.
        """
        try:
            closed_trades = self.broker_client.get_closed_trades()
            # Agregar solo los nuevos trades que no estén ya en el historial
            # Usar strategy_name en lugar de magic_number ya que TradeRecord no lo tiene
            existing_trades = {
                (t.entry_time, t.exit_time, t.symbol, t.strategy_name) 
                for t in self.trade_history
            }
            supports_native_close_events = hasattr(self.broker_client, "consume_closed_position_events")
            for trade in closed_trades:
                trade_key = (trade.entry_time, trade.exit_time, trade.symbol, trade.strategy_name)
                if trade_key not in existing_trades:
                    self.trade_history.append(trade)
                    logger.debug("Trade cerrado agregado al historial: %s", trade)
                    if not supports_native_close_events:
                        close_key = (
                            trade.symbol,
                            trade.exit_time.isoformat(),
                            round(float(trade.exit_price), 8),
                            round(float(trade.size), 6),
                        )
                        if close_key not in self._emitted_close_trade_keys:
                            self._emitted_close_trade_keys.add(close_key)
                            ts_exit = trade.exit_time
                            if ts_exit.tzinfo is None:
                                ts_exit = ts_exit.replace(tzinfo=timezone.utc)
                            ts_event = ts_exit.isoformat()
                            size_abs = abs(float(trade.size))
                            self.order_executor._emit_event(
                                {
                                    "event_type": "order_fill",
                                    "ts_event": ts_event,
                                    "bar_index": -1,
                                    "symbol": trade.symbol,
                                    "timeframe": "",
                                    "side": "flat",
                                    "price_ref": "close",
                                    "price": float(trade.exit_price),
                                    "size": size_abs,
                                    "commission": 0.0,
                                    "slippage": 0.0,
                                    "reason": "close_trade_history",
                                    "order_id": "",
                                }
                            )
                            self.order_executor._emit_event(
                                {
                                    "event_type": "position",
                                    "ts_event": ts_event,
                                    "bar_index": -1,
                                    "symbol": trade.symbol,
                                    "timeframe": "",
                                    "side": "flat",
                                    "price_ref": "close",
                                    "price": float(trade.exit_price),
                                    "size": 0.0,
                                    "commission": 0.0,
                                    "slippage": 0.0,
                                    "reason": "position_close_trade_history",
                                    "order_id": "",
                                }
                            )
        except NotImplementedError:
            # El broker simulado no implementa get_closed_trades
            logger.debug("Broker no soporta get_closed_trades, historial no actualizado")
        except Exception as e:
            logger.error("Error actualizando historial de trades: %s", e)


