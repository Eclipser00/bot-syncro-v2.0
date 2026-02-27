"""Punto de entrada principal del bot de trading."""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from types import SimpleNamespace

import pandas as pd

# Asegura que la raiz del repo este en sys.path para importar config correctamente.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings, validate_config
from bot_trading.application.engine.bot_engine import TradingBot
from bot_trading.application.engine.order_executor import OrderExecutor
from bot_trading.application.risk_management import RiskManager
from bot_trading.application.strategies.pivot_zone_test_strategy import PivotZoneTestStrategy
from bot_trading.domain.entities import (
    AccountInfo,
    OrderResult,
    Position,
    PendingOrder,
    RiskLimits,
    SymbolConfig,
    TradeRecord,
)
from bot_trading.infrastructure.data_fetcher import (
    DevelopmentCsvDataProvider,
    MarketDataService,
)
from bot_trading.infrastructure.mt5_client import MetaTrader5Client, MT5ConnectionError
from bot_trading.visualization.live_plot_service import AutoVisualizerService, resolve_bot_events_path

logger = logging.getLogger(__name__)


class FakeBroker:
    """Broker simulado para ejecutar el bot sin conexiones reales."""

    def __init__(self) -> None:
        self.orders_sent: list = []
        self.open_positions: list = []
        self.closed_trades: list = []
        self.pending_orders: list[PendingOrder] = []
        # Eventos de cierre (SL/TP) para instrumentación de paridad.
        self.closed_position_events: list[dict] = []
        self.pending_market_orders: list = []
        self.filled_market_orders: list = []
        # Cierres diferidos (paridad con Backtrader).
        self.pending_close_positions: list[dict] = []
        self._next_order_id: int = 1
        self._last_price: dict[str, float] = {}
        # Info basica para sizing (tick_size y volumenes minimos)
        self._symbol_info: dict[str, SimpleNamespace] = {}

    def connect(self) -> None:
        logger.info("Simulando conexion a broker (FakeBroker)")

    def get_ohlcv(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Genera una serie OHLCV lineal para pruebas rápidas sin broker real."""
        index = pd.date_range(start=start, end=end, freq="1min")
        data = {
            "open": [1.0] * len(index),
            "high": [1.0] * len(index),
            "low": [1.0] * len(index),
            "close": [float(i) for i in range(len(index))],
            "volume": [1] * len(index),
        }
        df = pd.DataFrame(data, index=index)
        df.attrs["symbol"] = symbol
        return df

    def send_market_order(self, order_request):
        """Simula envio de ordenes y gestiona posiciones abiertas/cerradas."""
        self.orders_sent.append(order_request)
        order_id = self._next_order_id
        self._next_order_id += 1
        logger.info(
            "Orden simulada enviada: %s (ID: %d, Magic: %s)",
            order_request,
            order_id,
            getattr(order_request, "magic_number", None),
        )

        if order_request.order_type in {"BUY", "SELL"} and os.getenv("PARITY_FILL_DELAY", "0") == "1":
            self.pending_market_orders.append(SimpleNamespace(order_id=order_id, request=order_request))
            logger.info(
                "Orden market diferida (paridad): %s %s vol=%s (ID: %d)",
                order_request.order_type,
                order_request.symbol,
                order_request.volume,
                order_id,
            )
            return OrderResult(success=True, order_id=order_id, fill_price=None)

        fill_price: float | None = None
        if order_request.order_type in {"BUY", "SELL"}:
            entry_price = float(self._last_price.get(order_request.symbol, 0.0))
            fill_price = entry_price
            position = Position(
                symbol=order_request.symbol,
                volume=order_request.volume,
                entry_price=entry_price,
                stop_loss=order_request.stop_loss,
                take_profit=order_request.take_profit,
                strategy_name=order_request.comment or "unknown",
                open_time=datetime.now(timezone.utc),
                magic_number=order_request.magic_number,
            )
            position.direction = 1 if order_request.order_type == "BUY" else -1  # type: ignore[attr-defined]
            # Paridad Backtrader: los brackets de salida no se evalúan en la primera vela tras el fill.
            setattr(position, "_sltp_warmup_bars", 1)
            self.open_positions.append(position)
            logger.info("Posicion simulada abierta: %s %s (Magic: %s)", order_request.order_type, order_request.symbol, order_request.magic_number)

        elif order_request.order_type == "CLOSE":
            if order_request.magic_number is not None:
                positions_to_close = [
                    p for p in self.open_positions if p.symbol == order_request.symbol and p.magic_number == order_request.magic_number
                ]
                if not positions_to_close:
                    logger.warning("No se encontraron posiciones abiertas para cerrar: %s (Magic: %s)", order_request.symbol, order_request.magic_number)
            else:
                positions_to_close = [p for p in self.open_positions if p.symbol == order_request.symbol]
                logger.debug("Cerrando todas las posiciones de %s (sin Magic Number especificado)", order_request.symbol)

            for pos in positions_to_close:
                self.open_positions.remove(pos)
                # Precio de salida: usa último precio conocido, luego precio del request, luego entry_price
                exit_price = float(self._last_price.get(pos.symbol, getattr(order_request, "price", pos.entry_price)))
                fill_price = exit_price
                direction = getattr(pos, "direction", 1)
                pnl = (exit_price - float(pos.entry_price)) * float(direction) * float(pos.volume)
                trade_record = TradeRecord(
                    symbol=pos.symbol,
                    strategy_name=pos.strategy_name,
                    entry_time=pos.open_time,
                    exit_time=datetime.now(timezone.utc),
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    size=pos.volume,
                    pnl=pnl,
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                )
                self.closed_trades.append(trade_record)
                logger.info(
                    "TRADE_CLOSED CLOSE_CMD %s dir=%s size=%.4f entry=%.5f exit=%.5f pnl=%.2f magic=%s strategy=%s entry_time=%s",
                    pos.symbol,
                    "BUY" if direction > 0 else "SELL",
                    float(pos.volume),
                    float(pos.entry_price),
                    exit_price,
                    pnl,
                    pos.magic_number,
                    pos.strategy_name,
                    pos.open_time.isoformat(),
                )

        return OrderResult(success=True, order_id=order_id, fill_price=fill_price)

    def process_price_tick(self, symbol: str, close: float, high: float | None = None, low: float | None = None) -> None:
        """Actualiza el último precio conocido para fills deterministas en FakeBroker."""
        try:
            self._last_price[symbol] = float(close)
        except Exception:
            pass

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
    ) -> OrderResult:
        order_id = self._next_order_id
        self._next_order_id += 1
        po = PendingOrder(
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
        logger.info("Orden pendiente simulada creada: %s %s @ %.5f (ID: %d)", order_type, symbol, price, order_id)
        return OrderResult(success=True, order_id=order_id)

    def cancel_order(self, order_id: int) -> bool:
        before = len(self.pending_orders)
        self.pending_orders = [po for po in self.pending_orders if po.order_id != order_id]
        canceled = len(self.pending_orders) < before
        if canceled:
            logger.info("Orden pendiente simulada cancelada: %s", order_id)
        return canceled

    def modify_order(
        self,
        order_id: int,
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> OrderResult:
        for po in self.pending_orders:
            if po.order_id == order_id:
                if price is not None:
                    po.price = price
                if stop_loss is not None:
                    po.stop_loss = stop_loss
                if take_profit is not None:
                    po.take_profit = take_profit
                logger.info("Orden pendiente simulada modificada: %s -> price=%s sl=%s tp=%s", order_id, price, stop_loss, take_profit)
                return OrderResult(success=True, order_id=order_id)
        return OrderResult(success=False, order_id=None, error_message="not_found")

    def get_open_orders(self, symbol=None, magic_number=None):
        def _match(po: PendingOrder) -> bool:
            if symbol is not None and po.symbol != symbol:
                return False
            if magic_number is not None and po.magic_number != magic_number:
                return False
            return True

        return [po for po in self.pending_orders if _match(po)]

    def get_open_positions(self):
        return self.open_positions.copy()

    def get_closed_trades(self):
        return self.closed_trades.copy()

    def get_account_info(self):
        return AccountInfo(
            balance=20000.0,
            equity=20000.0,
            margin=0.0,
            margin_free=20000.0,
            margin_level=None,
        )

    def _finalize_close(
        self,
        pos: Position,
        exit_price: float,
        reason: str,
        now: datetime | None = None,
        *,
        position_side: str = "flat",
        position_size: float = 0.0,
    ) -> None:
        """Cierra una posición y registra trade/evento con el precio de salida indicado."""
        if pos not in self.open_positions:
            return
        self.open_positions.remove(pos)

        entry_price = float(getattr(pos, "entry_price", exit_price))
        direction = getattr(pos, "direction", 1)
        pnl = (exit_price - entry_price) * float(direction) * float(pos.volume)
        trade_record = TradeRecord(
            symbol=pos.symbol,
            strategy_name=pos.strategy_name,
            entry_time=pos.open_time,
            exit_time=now or datetime.now(timezone.utc),
            entry_price=entry_price,
            exit_price=exit_price,
            size=pos.volume,
            pnl=pnl,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
        )
        self.closed_trades.append(trade_record)
        fill_side = "short" if int(direction) > 0 else "long"
        try:
            self.closed_position_events.append({
                "symbol": pos.symbol,
                "volume": float(pos.volume),
                "direction": int(direction),
                "fill_side": fill_side,
                "exit_price": float(exit_price),
                "reason": str(reason),
                "position_side": str(position_side),
                "position_size": float(position_size),
                "magic_number": getattr(pos, "magic_number", None),
                "strategy_name": getattr(pos, "strategy_name", None),
            })
        except Exception:
            pass

        # Cancelar órdenes pendientes asociadas a esta posición.
        self.pending_orders = [
            po
            for po in self.pending_orders
            if not (po.symbol == pos.symbol and po.magic_number == pos.magic_number)
        ]
        logger.info(
            "TRADE_CLOSED %s %s dir=%s size=%.4f entry=%.5f exit=%.5f pnl=%.2f magic=%s strategy=%s entry_time=%s",
            reason,
            pos.symbol,
            "BUY" if direction > 0 else "SELL",
            float(pos.volume),
            entry_price,
            float(exit_price),
            pnl,
            pos.magic_number,
            pos.strategy_name,
            pos.open_time.isoformat(),
        )

    def _schedule_close(
        self,
        pos: Position,
        exit_price: float,
        reason: str,
    ) -> None:
        """Agenda un cierre para el siguiente tick (paridad con Backtrader)."""
        try:
            setattr(pos, "_pending_close", True)
        except Exception:
            pass
        self.pending_close_positions.append({
            "pos": pos,
            "exit_price": float(exit_price),
            "reason": str(reason),
        })
        logger.info(
            "TRADE_CLOSE_SCHEDULED %s %s dir=%s exit=%s magic=%s",
            reason,
            pos.symbol,
            "BUY" if getattr(pos, "direction", 1) > 0 else "SELL",
            float(exit_price),
            getattr(pos, "magic_number", None),
        )

    def _flush_pending_closes_for_symbol(
        self,
        symbol: str,
        now: datetime | None = None,
    ) -> None:
        if not self.pending_close_positions:
            return
        for item in list(self.pending_close_positions):
            pos = item.get("pos")
            if pos is None or getattr(pos, "symbol", None) != symbol:
                continue
            self.pending_close_positions.remove(item)
            try:
                delattr(pos, "_pending_close")
            except Exception:
                pass
            raw_exit = item.get("exit_price", None)
            if raw_exit is not None:
                exit_price = float(raw_exit)
            else:
                exit_price = float(self._last_price.get(symbol, getattr(pos, "entry_price", 0.0)))
            self._finalize_close(pos, float(exit_price), str(item.get("reason", "close")), now=now)

    def flush_pending_closes(self, now: datetime | None = None) -> None:
        """Cierra cualquier posición pendiente (por ejemplo, al finalizar el CSV)."""
        if not self.pending_close_positions:
            return
        for item in list(self.pending_close_positions):
            pos = item.get("pos")
            if pos is None:
                self.pending_close_positions.remove(item)
                continue
            self.pending_close_positions.remove(item)
            try:
                delattr(pos, "_pending_close")
            except Exception:
                pass
            symbol = getattr(pos, "symbol", "")
            raw_exit = item.get("exit_price", None)
            if raw_exit is not None:
                exit_price = float(raw_exit)
            else:
                exit_price = float(self._last_price.get(symbol, getattr(pos, "entry_price", 0.0)))
            self._finalize_close(pos, float(exit_price), str(item.get("reason", "close")), now=now)

    def process_price_tick(
        self,
        symbol: str,
        price: float,
        high: float | None = None,
        low: float | None = None,
        open_price: float | None = None,
    ) -> None:
        """Simula fills de SL/TP en modo paper usando el OHLC más reciente.

        Usa high/low si se proporcionan para detectar toques intrabar; fallback al close.
        """
        self._last_price[symbol] = float(price)
        price_close = float(price)
        open_f = float(open_price) if open_price is not None else price_close
        close_only = os.getenv("PARITY_CLOSE_ON_CLOSE", "0") == "1"
        if close_only:
            open_f = price_close
            high_f = price_close
            low_f = price_close
        else:
            high_f = float(high) if high is not None else price_close
            low_f = float(low) if low is not None else price_close
        delay_close = os.getenv("PARITY_CLOSE_DELAY", "0") == "1"

        # Cierra pendientes (si existen) al comienzo de la vela actual.
        # Para cierres intrabar diferidos, usa el open actual como precio de fill.
        self._flush_pending_closes_for_symbol(symbol)

        if self.pending_market_orders:
            for pm in list(self.pending_market_orders):
                req = getattr(pm, "request", None)
                if req is None or req.symbol != symbol:
                    continue
                entry_price = price_close
                position = Position(
                    symbol=req.symbol,
                    volume=req.volume,
                    entry_price=entry_price,
                    stop_loss=req.stop_loss,
                    take_profit=req.take_profit,
                    strategy_name=req.comment or "unknown",
                    open_time=datetime.now(timezone.utc),
                    magic_number=req.magic_number,
                )
                position.direction = 1 if req.order_type == "BUY" else -1  # type: ignore[attr-defined]
                setattr(position, "_sltp_warmup_bars", 1)
                self.open_positions.append(position)
                self.filled_market_orders.append(
                    SimpleNamespace(order_id=getattr(pm, "order_id", None), request=req, price=entry_price)
                )
                self.pending_market_orders.remove(pm)
                logger.info(
                    "Posicion simulada abierta (diferida): %s %s (Magic: %s)",
                    req.order_type,
                    req.symbol,
                    req.magic_number,
                )

        for pos in list(self.open_positions):
            if pos.symbol != symbol:
                continue
            if getattr(pos, "_pending_close", False):
                continue
            warmup_bars = int(getattr(pos, "_sltp_warmup_bars", 0) or 0)
            if warmup_bars > 0:
                setattr(pos, "_sltp_warmup_bars", warmup_bars - 1)
                continue

            direction = int(getattr(pos, "direction", 1))
            sl = pos.stop_loss
            tp = pos.take_profit

            tp_hit = False
            sl_hit = False
            tp_price: float | None = None
            sl_price: float | None = None

            if direction > 0:
                if tp is not None:
                    tp_f = float(tp)
                    if open_f >= tp_f:
                        tp_hit = True
                        tp_price = float(open_f)
                    elif high_f >= tp_f:
                        tp_hit = True
                        tp_price = tp_f
                if sl is not None:
                    sl_f = float(sl)
                    if open_f <= sl_f:
                        sl_hit = True
                        sl_price = float(open_f)
                    elif low_f <= sl_f:
                        sl_hit = True
                        sl_price = sl_f
            else:
                if tp is not None:
                    tp_f = float(tp)
                    if open_f <= tp_f:
                        tp_hit = True
                        tp_price = float(open_f)
                    elif low_f <= tp_f:
                        tp_hit = True
                        tp_price = tp_f
                if sl is not None:
                    sl_f = float(sl)
                    if open_f >= sl_f:
                        sl_hit = True
                        sl_price = float(open_f)
                    elif high_f >= sl_f:
                        sl_hit = True
                        sl_price = sl_f

            if not tp_hit and not sl_hit:
                continue

            both_sides_hit = tp_hit and sl_hit
            secondary_price: float | None = None
            if both_sides_hit:
                # Backtrader puede ejecutar ambas órdenes opuestas en la misma vela.
                # Priorizamos SL como primer fill para alinear el orden observado.
                hit_reason = "SL"
                exit_price = float(sl_price if sl_price is not None else tp_price if tp_price is not None else price_close)
                secondary_price = float(tp_price if tp_price is not None else exit_price)
            elif tp_hit:
                hit_reason = "TP"
                exit_price = float(tp_price if tp_price is not None else price_close)
            else:
                hit_reason = "SL"
                exit_price = float(sl_price if sl_price is not None else price_close)

            if delay_close:
                self._schedule_close(pos, float(exit_price), str(hit_reason))
                continue

            if both_sides_hit:
                reverse_direction = -1 if direction > 0 else 1
                reverse_side = "long" if reverse_direction > 0 else "short"
                self._finalize_close(
                    pos,
                    float(exit_price),
                    str(hit_reason),
                    position_side=reverse_side,
                    position_size=float(pos.volume),
                )
                reverse_entry = float(secondary_price if secondary_price is not None else exit_price)
                reverse_pos = Position(
                    symbol=pos.symbol,
                    volume=pos.volume,
                    entry_price=reverse_entry,
                    stop_loss=None,
                    take_profit=None,
                    strategy_name=pos.strategy_name,
                    open_time=datetime.now(timezone.utc),
                    magic_number=pos.magic_number,
                )
                reverse_pos.direction = reverse_direction  # type: ignore[attr-defined]
                setattr(reverse_pos, "_parity_reversal", True)
                setattr(reverse_pos, "_sltp_warmup_bars", 1)
                self.open_positions.append(reverse_pos)
                continue

            self._finalize_close(pos, float(exit_price), str(hit_reason))

    def _get_symbol_info(self, symbol: str) -> SimpleNamespace:
        """Devuelve info basica del simbolo para calcular lotes en modo paper."""
        if symbol not in self._symbol_info:
            self._symbol_info[symbol] = SimpleNamespace(
                trade_tick_value=1.0,
                trade_tick_size=0.0001,
                volume_min=0.01,
                volume_step=0.01,
                volume_max=10.0,
            )
        return self._symbol_info[symbol]

    def consume_filled_market_orders(self) -> list:
        filled = list(self.filled_market_orders)
        self.filled_market_orders.clear()
        return filled

    def consume_closed_position_events(self) -> list[dict]:
        events = list(self.closed_position_events)
        self.closed_position_events.clear()
        return events


def _configure_logging(logging_cfg) -> None:
    """Configura logging segun config.py (consola y opcional a archivo)."""
    level = getattr(logging, str(logging_cfg.level).upper(), logging.INFO)
    kwargs = {"level": level, "format": logging_cfg.format, "force": True}

    if logging_cfg.log_to_file and logging_cfg.log_file_path:
        log_path = (ROOT / logging_cfg.log_file_path).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            log_path.unlink(missing_ok=True)
        except Exception:
            pass
        kwargs["handlers"] = [logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]

    logging.basicConfig(**kwargs)
    pivot_log_path = None
    pivot_logger = logging.getLogger("pivot_zones")
    pivot_logger.setLevel(logging.INFO)
    pivot_logger.handlers.clear()
    pivot_logger.propagate = False

    if logging_cfg.pivot_log_file_path:
        pivot_log_path = (ROOT / logging_cfg.pivot_log_file_path).resolve()
        pivot_log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            pivot_log_path.unlink(missing_ok=True)
        except Exception:
            pass

        pivot_handler = logging.FileHandler(pivot_log_path)
        pivot_handler.setLevel(logging.INFO)
        # Usamos timestamp de los datos en el propio mensaje; evitamos timestamp del runtime aquí.
        pivot_handler.setFormatter(logging.Formatter("%(message)s"))
        pivot_logger.addHandler(pivot_handler)

    logger.info(
        "Logging configurado (level=%s, file=%s, pivot_log=%s)",
        logging_cfg.level,
        logging_cfg.log_file_path,
        pivot_log_path,
    )


def _build_broker():
    """Crea broker real o simulado en funcion de config.py."""
    broker_cfg = settings.broker
    if broker_cfg.use_real_broker:
        logger.info("Modo PRODUCCION: MetaTrader5 (retries=%d, delay=%.2fs)", broker_cfg.max_retries, broker_cfg.retry_delay)
        broker = MetaTrader5Client(max_retries=broker_cfg.max_retries, retry_delay=broker_cfg.retry_delay)
        try:
            broker.connect()
            logger.info("Conexion exitosa con MetaTrader5")
        except MT5ConnectionError as e:
            logger.error("Error al conectar con MetaTrader5: %s", e)
            raise
        except Exception as e:
            logger.error("Error inesperado al conectar con MetaTrader5: %s", e)
            raise
        return broker

    logger.info("Modo DESARROLLO: FakeBroker (simulado)")
    broker = FakeBroker()
    broker.connect()
    return broker


def _build_market_data_service(broker) -> MarketDataService:
    """Inicializa el servicio de datos segun el modo configurado."""
    data_cfg = settings.data
    lookback_entry = data_cfg.bootstrap_lookback_days_entry
    lookback_zone = data_cfg.bootstrap_lookback_days_zone
    lookback_stop = data_cfg.bootstrap_lookback_days_stop

    if data_cfg.data_mode == "production":
        logger.info("Proveedor de datos: broker real (production)")
        provider = broker
    else:
        data_dir = (ROOT / data_cfg.data_development_dir).resolve()
        logger.info("Proveedor de datos: CSV de desarrollo en %s", data_dir)
        provider = DevelopmentCsvDataProvider(
            data_dir=data_dir,
            base_timeframe=data_cfg.csv_base_timeframe,
            lookback_days_entry=lookback_entry,
            lookback_days_zone=lookback_zone,
            lookback_days_stop=lookback_stop,
        )

    return MarketDataService(
        data_provider=provider,
        lookback_days_entry=lookback_entry,
        lookback_days_zone=lookback_zone,
        lookback_days_stop=lookback_stop,
    )


def _build_risk_manager() -> RiskManager:
    """Construye el RiskManager a partir de config.py."""
    risk_cfg = settings.risk
    risk_limits = RiskLimits(
        dd_global=risk_cfg.dd_global,
        dd_por_activo=risk_cfg.dd_por_activo,
        dd_por_estrategia=risk_cfg.dd_por_estrategia,
        initial_balance=risk_cfg.initial_balance,
        max_margin_usage_percent=risk_cfg.max_margin_usage_percent,
    )
    return RiskManager(risk_limits)


def _build_symbols() -> List[SymbolConfig]:
    """Mapea configuracion declarativa de simbolos a objetos de dominio."""
    symbols: List[SymbolConfig] = []
    for sym_cfg in settings.symbols:
        symbols.append(
            SymbolConfig(
                name=sym_cfg.name,
                min_timeframe=sym_cfg.min_timeframe,
                n1=getattr(sym_cfg, "n1", None),
                n2=getattr(sym_cfg, "n2", None),
                n3=getattr(sym_cfg, "n3", None),
                size_pct=getattr(sym_cfg, "size_pct", None),
                p=getattr(sym_cfg, "p", None),
            )
        )
    return symbols

def _collect_symbol_params() -> dict[str, dict[str, float]]:
    """Extrae overrides por símbolo para estrategias desde config.py."""
    params: dict[str, dict[str, float]] = {}
    for sym_cfg in settings.symbols:
        overrides: dict[str, float] = {}
        for field_name in ("n1", "n2", "n3", "size_pct", "p"):
            value = getattr(sym_cfg, field_name, None)
            if value is not None:
                if field_name in {"n1", "n2", "n3"}:
                    overrides[field_name] = int(value)
                else:
                    overrides[field_name] = float(value)
        if overrides:
            params[sym_cfg.name] = overrides
    return params


def _build_strategies(broker) -> List[PivotZoneTestStrategy]:
    """Crea instancias de estrategias declaradas en config.py."""
    strategies: List[PivotZoneTestStrategy] = []
    symbol_params = _collect_symbol_params()
    allowed_all = [s.name for s in settings.symbols]
    for strat_cfg in settings.strategies:
        allowed_symbols = list(strat_cfg.allowed_symbols) if strat_cfg.allowed_symbols else list(allowed_all)
        strategies.append(
            PivotZoneTestStrategy(
                name=strat_cfg.name,
                timeframes=list(strat_cfg.timeframes),
                broker_client=broker,
                allowed_symbols=allowed_symbols,
                tf_entry=strat_cfg.tf_entry,
                tf_zone=strat_cfg.tf_zone,
                tf_stop=strat_cfg.tf_stop,
                n1=strat_cfg.n1,
                n2=strat_cfg.n2,
                n3=strat_cfg.n3,
                size_pct=strat_cfg.size_pct,
                p=strat_cfg.p,
                symbol_params=symbol_params,
            )
        )
    return strategies


def _build_auto_visualizer(symbols: List[SymbolConfig]) -> AutoVisualizerService | None:
    """Inicializa el visualizador automatico integrado al loop principal.

    Se habilita por defecto en todos los modos cuando se arranca el bot normal
    (`python config.py` / `python -m bot_trading.main`).
    Puede deshabilitarse con `VISUALIZER_AUTO_ENABLE=0`.
    """
    force_enable = os.getenv("VISUALIZER_AUTO_ENABLE")
    if force_enable is None:
        enable = True
    else:
        enable = force_enable.strip() not in {"0", "false", "False", ""}

    if not enable:
        logger.info("AutoVisualizer deshabilitado")
        return None

    refresh_env = os.getenv("VISUALIZER_REFRESH_SECONDS")
    if refresh_env:
        try:
            refresh_seconds = max(1, int(refresh_env))
        except Exception:
            refresh_seconds = 900 if settings.mode == "development" else 900
    else:
        refresh_seconds = 900 if settings.mode == "development" else 900

    output_dir = (ROOT / "outputs").resolve()
    pivot_log_path = (ROOT / (settings.logging.pivot_log_file_path or "logs/pivot_zones.log")).resolve()
    bot_events_path = resolve_bot_events_path(ROOT, "outputs/bot_events.jsonl")

    try:
        return AutoVisualizerService(
            symbols=symbols,
            output_dir=output_dir,
            pivot_log_path=pivot_log_path,
            bot_events_path=bot_events_path,
            refresh_seconds=refresh_seconds,
            start_from_end=True,
        )
    except Exception as exc:
        logger.error("No se pudo inicializar AutoVisualizer: %s", exc, exc_info=True)
        return None


def _log_final_stats(broker) -> None:
    """Imprime estadisticas finales segun el broker activo."""
    logger.info("=" * 80)
    logger.info("ESTADISTICAS FINALES")
    logger.info("=" * 80)

    if settings.broker.use_real_broker:
        try:
            positions = broker.get_open_positions()
            logger.info("Posiciones abiertas: %d", len(positions))
            for pos in positions:
                logger.info(
                    "- %s: %.2f lotes @ %.5f (Strategy: %s, Magic: %s)",
                    pos.symbol,
                    pos.volume,
                    pos.entry_price,
                    pos.strategy_name,
                    pos.magic_number,
                )

            trades = broker.get_closed_trades()
            logger.info("Trades cerrados hoy: %d", len(trades))
            if trades:
                total_pnl = sum(t.pnl for t in trades)
                logger.info("PnL total: %.2f", total_pnl)
                logger.info("Ultimos trades (max 5):")
                for trade in trades[:5]:
                    logger.info(
                        "- %s: %.2f lotes, PnL=%.2f, Entrada=%.5f, Salida=%.5f",
                        trade.symbol,
                        trade.size,
                        trade.pnl,
                        trade.entry_price,
                        trade.exit_price,
                    )

        except Exception as e:
            logger.error("Error al obtener estadisticas finales: %s", e)
    else:
        logger.info("Ordenes simuladas enviadas: %d", len(getattr(broker, 'orders_sent', [])))
        logger.info("Posiciones simuladas abiertas: %d", len(getattr(broker, 'open_positions', [])))
        closed = getattr(broker, 'closed_trades', [])
        logger.info("Trades simulados cerrados: %d", len(closed))
        pnl_total = sum(getattr(t, "pnl", 0.0) or 0.0 for t in closed)
        logger.info("PNL simulado acumulado: %.2f", pnl_total)
        for trade in closed[:10]:  # limitar log
            logger.debug(
                "- %s entry=%.5f exit=%.5f pnl=%.2f strategy=%s magic=%s",
                trade.symbol,
                getattr(trade, "entry_price", None),
                getattr(trade, "exit_price", None),
                getattr(trade, "pnl", None),
                trade.strategy_name,
                getattr(trade, "magic_number", None),
            )
        # Cerrar posiciones abiertas al agotar datos (precio último conocido)
        for pos in list(getattr(broker, 'open_positions', [])):
            last_price = getattr(getattr(broker, '_last_price', {}), 'get', lambda *_: getattr(pos, "entry_price", 0.0))(pos.symbol)
            broker.send_market_order(
                SimpleNamespace(
                    symbol=pos.symbol,
                    order_type="CLOSE",
                    volume=pos.volume,
                    price=last_price,
                    stop_loss=None,
                    take_profit=None,
                    magic_number=getattr(pos, "magic_number", None),
                    comment=pos.strategy_name,
                )
            )
        if getattr(broker, 'open_positions', []):
            logger.info("Posiciones simuladas cerradas al agotar datos: %d", len(getattr(broker, 'open_positions', [])))

    if settings.broker.use_real_broker:
        logger.info("Cerrando conexion con MetaTrader5...")
        logger.info("Conexion cerrada")

    logger.info("=" * 80)
    logger.info("Bot finalizado")
    logger.info("=" * 80)


def main() -> None:
    """Ejecuta el bot de trading leyendo toda la configuracion desde config.py."""
    validate_config(settings)
    _configure_logging(settings.logging)

    logger.info("=" * 80)
    logger.info("Iniciando Bot de Trading (%s) en modo %s", settings.name, settings.mode)
    logger.info("=" * 80)

    broker = _build_broker()
    market_data_service = _build_market_data_service(broker)
    risk_manager = _build_risk_manager()
    order_executor = OrderExecutor(broker)
    strategies = _build_strategies(broker)
    symbols = _build_symbols()
    auto_visualizer = _build_auto_visualizer(symbols)

    bot = TradingBot(
        broker_client=broker,
        market_data_service=market_data_service,
        risk_manager=risk_manager,
        order_executor=order_executor,
        strategies=strategies,
        symbols=symbols,
        market_data_callback=(auto_visualizer.on_market_data if auto_visualizer is not None else None),
    )
    logger.info("Bot inicializado correctamente")

    logger.info("=" * 80)
    logger.info("Modo de ejecucion: bucle sincronizado")
    logger.info("Timeframe base: %s minutos; espera post-cierre: %s segundos", settings.loop.timeframe_minutes, settings.loop.wait_after_close)
    logger.info("Presiona Ctrl+C para detener el bot")
    logger.info("=" * 80)

    try:
        bot.run_synchronized(
            timeframe_minutes=settings.loop.timeframe_minutes,
            wait_after_close=settings.loop.wait_after_close,
            skip_sleep_when_simulated=settings.loop.skip_sleep_when_simulated,
        )
    except KeyboardInterrupt:
        logger.info("Ejecucion interrumpida por el usuario")
    except Exception as e:
        logger.error("Error durante la ejecucion del bot: %s", e, exc_info=True)
        raise
    finally:
        if auto_visualizer is not None:
            auto_visualizer.close()
        _log_final_stats(broker)


if __name__ == "__main__":
    main()
