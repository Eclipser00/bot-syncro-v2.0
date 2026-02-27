"""
Runner determinista para reproducir el bot en modo desarrollo (CSV) y generar
`outputs/bot_events.jsonl` para la comparación de paridad.

Uso:
    python -m bot_trading.replay_runner
"""
import os
from config import ACTIVE_ENV
from bot_trading import main


def run() -> None:
    # Forzamos entorno development por si el usuario cambió ACTIVE_ENV.
    os.environ["ACTIVE_ENV"] = "development"
    # Ejecuta el flujo estándar (usa DevelopmentCsvDataProvider + FakeBroker).
    main.main()


if __name__ == "__main__":
    run()
