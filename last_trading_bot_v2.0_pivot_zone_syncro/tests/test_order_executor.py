"""Tests del ejecutor de órdenes."""
from bot_trading.application.engine.order_executor import OrderExecutor
from bot_trading.domain.entities import OrderRequest, OrderResult


class FakeBroker:
    """Broker simulado que almacena la última orden enviada."""

    def __init__(self) -> None:
        self.last_order: OrderRequest | None = None

    def send_market_order(self, order_request: OrderRequest) -> OrderResult:
        self.last_order = order_request
        return OrderResult(success=True, order_id=1)

    def get_open_positions(self):
        return []


def test_order_executor_envia_orden_y_interpreta_respuesta() -> None:
    """El ejecutor debe enviar la orden y retornar el resultado del broker."""
    broker = FakeBroker()
    executor = OrderExecutor(broker)
    request = OrderRequest(symbol="EURUSD", volume=0.01, order_type="BUY")

    result = executor.execute_order(request)

    assert broker.last_order == request
    assert result.success is True
    assert result.order_id == 1


def test_order_executor_eventos_fill_y_position_usan_fill_price_real() -> None:
    class BrokerConFill:
        def send_market_order(self, order_request: OrderRequest) -> OrderResult:
            return OrderResult(success=True, order_id=7, fill_price=1.23456)

        def get_open_positions(self):
            return []

    class ExecutorSpy(OrderExecutor):
        def __init__(self, broker_client) -> None:
            super().__init__(broker_client)
            self.events: list[dict] = []

        def _emit_event(self, payload: dict) -> None:  # type: ignore[override]
            self.events.append(payload)

    broker = BrokerConFill()
    executor = ExecutorSpy(broker)
    request = OrderRequest(symbol="EURUSD", volume=0.01, order_type="BUY", timeframe="M1")

    result = executor.execute_order(request)

    assert result.success is True
    fill_events = [e for e in executor.events if e.get("event_type") == "order_fill"]
    position_events = [e for e in executor.events if e.get("event_type") == "position"]
    assert fill_events
    assert position_events
    assert float(fill_events[-1]["price"]) == 1.23456
    assert float(position_events[-1]["price"]) == 1.23456
