"""Entrypoint CLI para el visualizador M3."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import ACTIVE_ENV
from bot_trading.visualization.runner import build_runner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualizador dinamico M3 (Bokeh)")
    parser.add_argument("--mode", default=ACTIVE_ENV, choices=["development", "production"], help="Modo de ejecucion")
    parser.add_argument("--refresh-seconds", type=int, default=900, help="Intervalo de refresco del HTML (segundos)")
    parser.add_argument(
        "--cycle-seconds",
        type=int,
        default=None,
        help="Sleep entre ciclos de ingesta (segundos). Por defecto: 0 dev, 60 prod.",
    )
    parser.add_argument("--output-html", default="outputs/visualizer_m3.html", help="Ruta del HTML de salida")
    parser.add_argument("--pivot-log", default="logs/pivot_zones.log", help="Ruta del log de zonas pivote")
    parser.add_argument("--bot-events", default="outputs/bot_events.jsonl", help="Ruta del JSONL de eventos del bot")
    parser.add_argument(
        "--max-candles",
        type=int,
        default=0,
        help="Maximo de velas M3 en memoria por simbolo (0 o negativo = sin limite)",
    )
    parser.add_argument("--max-events", type=int, default=1500, help="Maximo de eventos en memoria por simbolo")
    parser.add_argument("--max-cycles", type=int, default=None, help="Maximo de ciclos a ejecutar (opcional)")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logging.getLogger("bokeh").setLevel(logging.WARNING)

    max_candles = None if args.max_candles is None or int(args.max_candles) <= 0 else int(args.max_candles)

    runner = build_runner(
        mode=args.mode,
        refresh_seconds=args.refresh_seconds,
        cycle_sleep_seconds=args.cycle_seconds,
        output_html=args.output_html,
        pivot_log=args.pivot_log,
        bot_events=args.bot_events,
        max_candles=max_candles,
        max_events=args.max_events,
        max_cycles=args.max_cycles,
    )
    logging.getLogger(__name__).info("Salida HTML: %s", Path(args.output_html).as_posix())
    runner.run()


if __name__ == "__main__":
    main()
