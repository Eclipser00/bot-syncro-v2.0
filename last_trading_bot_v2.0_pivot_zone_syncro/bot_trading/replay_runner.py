"""
Runner determinista para reproducir el bot en modo desarrollo (CSV) y generar
`outputs/bot_events.jsonl` para la comparación de paridad.

Uso:
    python -m bot_trading.replay_runner
"""
import importlib

import config


def run() -> None:
    # Forzamos development sobre el modulo de configuracion antes de importar el entrypoint.
    config.ACTIVE_ENV = "development"
    config.settings = config.load_settings("development")
    # Ejecuta el flujo estándar (usa DevelopmentCsvDataProvider + FakeBroker).
    bot_main = importlib.import_module("bot_trading.main")
    bot_main.main()


if __name__ == "__main__":
    run()
