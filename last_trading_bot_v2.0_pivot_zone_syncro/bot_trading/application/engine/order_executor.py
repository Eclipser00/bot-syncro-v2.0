"""Ejecución de órdenes a través del broker."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from bot_trading.domain.entities import OrderRequest, OrderResult, Position
from bot_trading.infrastructure.mt5_client import BrokerClient
from bot_trading.application.utils.event_logger import EventLogger

logger = logging.getLogger(__name__)
_EVENT_LOGGER = EventLogger(str(Path(__file__).resolve().parents[4] / "outputs" / "bot_events.jsonl"))


@dataclass
class OrderExecutor:
    """Encapsula la lógica de envío y verificación de órdenes."""

    broker_client: BrokerClient
    open_positions: dict[str, Position] = field(default_factory=dict)
    pending_market_fills: list[dict] = field(default_factory=list)

    def _emit_event(self, payload: dict) -> None:
        try:
            _EVENT_LOGGER.log(payload)
        except Exception:
            pass

    def _resolve_fill_price(self, order_request: OrderRequest, result: OrderResult) -> float:
        """Obtiene el mejor precio disponible para eventos de fill."""
        if result.fill_price is not None and float(result.fill_price) > 0.0:
            return float(result.fill_price)
        if hasattr(self.broker_client, "_last_price"):
            try:
                return float(getattr(self.broker_client, "_last_price", {}).get(order_request.symbol, 0.0))
            except Exception:
                return 0.0
        return 0.0

    def _generate_position_key(self, symbol: str, magic_number: int | None) -> str:
        """Genera una clave única para identificar posiciones.
        
        Args:
            symbol: Símbolo de la posición.
            magic_number: Magic Number de la estrategia.
            
        Returns:
            Clave única basada en símbolo y magic_number.
        """
        if magic_number is not None:
            return f"{symbol}_{magic_number}"
        return symbol

    def sync_state(self) -> None:
        """Sincroniza el estado interno con las posiciones reales del broker."""
        try:
            real_positions = self.broker_client.get_open_positions()
            # Reconstruimos el mapa local usando símbolo + magic_number como clave
            # Esto permite múltiples posiciones del mismo símbolo con diferentes estrategias
            self.open_positions.clear()
            for pos in real_positions:
                position_key = self._generate_position_key(pos.symbol, pos.magic_number)
                self.open_positions[position_key] = pos
            logger.debug("Estado sincronizado con broker. %d posiciones abiertas.", len(self.open_positions))
        except Exception as e:
            logger.error("Error sincronizando posiciones con broker: %s", e)

    def execute_order(self, order_request: OrderRequest) -> OrderResult:
        """Envía una orden al broker y devuelve el resultado."""
        logger.info(
            "Enviando orden %s de volumen %s para %s (Magic: %s)",
            order_request.order_type,
            order_request.volume,
            order_request.symbol,
            order_request.magic_number,
        )
        # Evento de env?o de orden (instrumentaci?n de paridad)
        self._emit_event({
            'event_type': 'order_submit',
            'ts_event': order_request.ts_event or datetime.now(timezone.utc).isoformat(),
            'bar_index': order_request.bar_index if order_request.bar_index is not None else -1,
            'symbol': order_request.symbol,
            'timeframe': order_request.timeframe if order_request.timeframe is not None else '',
            'side': 'long' if order_request.order_type == 'BUY' else 'short' if order_request.order_type == 'SELL' else 'flat',
            'price_ref': order_request.price_ref or 'market',
            'price': getattr(self.broker_client, '_last_price', {}).get(order_request.symbol, 0.0) if hasattr(self.broker_client, '_last_price') else 0.0,
            'size': float(order_request.volume),
            'commission': 0.0,
            'slippage': 0.0,
            'reason': 'submit',
            'order_id': str(getattr(order_request, 'magic_number', '')),
        })
        result = self.broker_client.send_market_order(order_request)
        if not result.success:
            logger.error("Orden rechazada: %s", result.error_message)
        else:
            logger.debug("Orden aceptada con id %s", result.order_id)
            fill_price = self._resolve_fill_price(order_request, result)
            delay_fill = os.getenv("PARITY_FILL_DELAY", "0") == "1" and hasattr(self.broker_client, "consume_filled_market_orders")
            if delay_fill and order_request.order_type in {"BUY", "SELL"}:
                self.pending_market_fills.append(
                    {
                        "order_request": order_request,
                        "order_id": getattr(result, "order_id", None),
                    }
                )
            else:
                self._emit_event({
                    'event_type': 'order_fill',
                    'ts_event': order_request.ts_event or datetime.now(timezone.utc).isoformat(),
                    'bar_index': order_request.bar_index if order_request.bar_index is not None else -1,
                    'symbol': order_request.symbol,
                    'timeframe': order_request.timeframe if order_request.timeframe is not None else '',
                    'side': 'long' if order_request.order_type == 'BUY' else 'short' if order_request.order_type == 'SELL' else 'flat',
                    'price_ref': order_request.price_ref or 'market',
                    'price': fill_price,
                    'size': float(order_request.volume),
                    'commission': 0.0,
                    'slippage': 0.0,
                    'reason': 'fill',
                    'order_id': str(getattr(result, 'order_id', '')),
                })

                if order_request.order_type in {"BUY", "SELL"}:
                    self._register_position(order_request, result, fill_price=fill_price)
                elif order_request.order_type == "CLOSE":
                    self._remove_position(
                        order_request.symbol,
                        order_request.magic_number,
                        closed_price=fill_price,
                        ts_event=order_request.ts_event,
                        bar_index=order_request.bar_index,
                        timeframe=order_request.timeframe,
                    )
                
        return result

    def flush_pending_fills(self, now: datetime | None = None) -> None:
        if not self.pending_market_fills:
            return
        if not hasattr(self.broker_client, "consume_filled_market_orders"):
            return
        filled = self.broker_client.consume_filled_market_orders()  # type: ignore[attr-defined]
        if not filled:
            return
        for fill in filled:
            order_id = getattr(fill, "order_id", None)
            order_req = None
            for item in list(self.pending_market_fills):
                if item.get("order_id") == order_id:
                    order_req = item.get("order_request")
                    self.pending_market_fills.remove(item)
                    break
            if order_req is None:
                continue
            ts_event = (now or datetime.now(timezone.utc)).isoformat()
            bar_index = order_req.bar_index + 1 if order_req.bar_index is not None else -1
            price_fill = getattr(fill, "price", None)
            price_val = float(price_fill) if price_fill is not None else getattr(self.broker_client, '_last_price', {}).get(order_req.symbol, 0.0)
            self._emit_event({
                'event_type': 'order_fill',
                'ts_event': ts_event,
                'bar_index': bar_index,
                'symbol': order_req.symbol,
                'timeframe': order_req.timeframe if order_req.timeframe is not None else '',
                'side': 'long' if order_req.order_type == 'BUY' else 'short' if order_req.order_type == 'SELL' else 'flat',
                'price_ref': order_req.price_ref or 'market',
                'price': price_val,
                'size': float(order_req.volume),
                'commission': 0.0,
                'slippage': 0.0,
                'reason': 'fill',
                'order_id': str(order_id or ''),
            })
            self._register_position(order_req, OrderResult(success=True, order_id=order_id, fill_price=price_val), fill_price=price_val)

    def _remove_position(
        self,
        symbol: str,
        magic_number: int | None = None,
        *,
        emit_event: bool = True,
        closed_price: float | None = None,
        ts_event: str | None = None,
        bar_index: int | None = None,
        timeframe: str | int | None = None,
    ) -> None:
        """Elimina posiciones internas asociadas a un símbolo y opcionalmente a una estrategia.
        
        Args:
            symbol: Símbolo de la posición a eliminar.
            magic_number: Si se proporciona, solo elimina posiciones con este Magic Number.
                         Si es None, elimina todas las posiciones del símbolo.
        """
        if magic_number is not None:
            # Eliminar posición específica de la estrategia
            position_key = self._generate_position_key(symbol, magic_number)
            if position_key in self.open_positions:
                del self.open_positions[position_key]
                logger.debug("Posición eliminada: %s con Magic Number: %s", symbol, magic_number)
            else:
                logger.warning("No se encontró posición para eliminar: %s (Magic: %s)", symbol, magic_number)
        else:
            # Fallback: eliminar todas las posiciones del símbolo
            keys_to_remove = [k for k, v in self.open_positions.items() if v.symbol == symbol]
            for k in keys_to_remove:
                del self.open_positions[k]
            logger.debug("Posiciones eliminadas del registro local para %s", symbol)

        if emit_event:
            price_value = (
                float(closed_price)
                if closed_price is not None and float(closed_price) > 0.0
                else (getattr(self.broker_client, '_last_price', {}).get(symbol, 0.0) if hasattr(self.broker_client, '_last_price') else 0.0)
            )
            self._emit_event({
                'event_type': 'position',
                'ts_event': ts_event or datetime.now(timezone.utc).isoformat(),
                'bar_index': bar_index if bar_index is not None else -1,
                'symbol': symbol,
                'timeframe': timeframe if timeframe is not None else (4 if getattr(self.broker_client, '_last_price', None) is not None else ''),
                'side': 'flat',
                'price_ref': 'market',
                'price': price_value,
                'size': 0.0,
                'commission': 0.0,
                'slippage': 0.0,
                'reason': 'position_close',
                'order_id': str(magic_number) if magic_number is not None else ''
            })

    def _register_position(self, order_request: OrderRequest, result: OrderResult, fill_price: float | None = None) -> None:
        """Registra una posición abierta.
        
        Args:
            order_request: Solicitud de orden que generó la posición.
            result: Resultado de la orden ejecutada.
        """
        position = Position(
            symbol=order_request.symbol,
            volume=order_request.volume,
            entry_price=0.0,  # Se actualizará con datos reales del broker
            stop_loss=order_request.stop_loss,
            take_profit=order_request.take_profit,
            strategy_name=order_request.comment or "unknown",
            open_time=datetime.now(timezone.utc),
            magic_number=order_request.magic_number,
        )
        # Usar símbolo + magic_number como clave para consistencia
        position_key = self._generate_position_key(order_request.symbol, order_request.magic_number)
        self.open_positions[position_key] = position
        logger.debug("Posición registrada: %s con Magic Number: %s", position_key, order_request.magic_number)
        event_price = (
            float(fill_price)
            if fill_price is not None and float(fill_price) > 0.0
            else (getattr(self.broker_client, '_last_price', {}).get(order_request.symbol, 0.0) if hasattr(self.broker_client, '_last_price') else 0.0)
        )
        self._emit_event({
            'event_type': 'position',
            'ts_event': order_request.ts_event or datetime.now(timezone.utc).isoformat(),
            'bar_index': order_request.bar_index if order_request.bar_index is not None else -1,
            'symbol': order_request.symbol,
            'timeframe': order_request.timeframe if order_request.timeframe is not None else '',
            'side': 'long' if order_request.order_type == 'BUY' else 'short',
            'price_ref': order_request.price_ref or 'market',
            'price': event_price,
            'size': float(order_request.volume),
            'commission': 0.0,
            'slippage': 0.0,
            'reason': 'position_open',
            'order_id': str(getattr(order_request, 'magic_number', '')),
        })

    def has_open_position(
        self, 
        symbol: str, 
        strategy_name: str = None,
        magic_number: int = None
    ) -> bool:
        """Verifica si existe una posición abierta para el símbolo dado.
        
        Args:
            symbol: Símbolo a verificar.
            strategy_name: Opcional, filtrar por nombre de estrategia (fallback legacy).
            magic_number: Opcional, filtrar por Magic Number (método preferido y robusto).
            
        Returns:
            True si existe al menos una posición abierta que coincida con los criterios.
        """
        # Método preferido: búsqueda directa usando clave
        if magic_number is not None:
            position_key = self._generate_position_key(symbol, magic_number)
            return position_key in self.open_positions
        
        # Fallback: búsqueda iterativa (menos eficiente, para compatibilidad)
        for pos in self.open_positions.values():
            if pos.symbol == symbol:
                if strategy_name is None or pos.strategy_name.startswith(strategy_name):
                    return True
        return False
