"""Runner del visualizador M3 incremental."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import ACTIVE_ENV, load_settings
from bot_trading.domain.entities import SymbolConfig

from .data_source import BaseCandleDataSource, DevelopmentCandleDataSource, ProductionCandleDataSource
from .event_reader import EventReader
from .plot_bokeh import BokehPlotBuilder
from .state_store import VisualizerStateStore

logger = logging.getLogger(__name__)


def _resolve_bot_events_path(root: Path, configured_path: str) -> Path:
    """Resuelve la ruta de eventos del bot priorizando la fuente más reciente.

    Soporta dos ubicaciones reales del proyecto:
    - `<repo>/outputs/bot_events.jsonl`
    - `<repo_parent>/outputs/bot_events.jsonl`
    """
    local_path = (root / configured_path).resolve()
    parent_path = (root.parent / configured_path).resolve()

    local_exists = local_path.exists()
    parent_exists = parent_path.exists()

    if local_exists and parent_exists:
        local_mtime = local_path.stat().st_mtime
        parent_mtime = parent_path.stat().st_mtime
        selected = parent_path if parent_mtime > local_mtime else local_path
        if selected == parent_path:
            logger.info("Eventos del bot detectados en ruta padre: %s", parent_path)
        return selected

    if parent_exists:
        logger.info("Eventos del bot detectados en ruta padre: %s", parent_path)
        return parent_path

    return local_path


@dataclass
class VisualizerRunner:
    """Orquesta ingesta incremental de velas/eventos y refresco del HTML."""

    mode: str
    data_source: BaseCandleDataSource
    event_reader: EventReader
    state_store: VisualizerStateStore
    plot_builder: BokehPlotBuilder
    refresh_seconds: int = 900
    cycle_sleep_seconds: int = 60
    max_cycles: int | None = None

    def run(self) -> None:
        logger.info(
            "Iniciando visualizador M3 (mode=%s, refresh=%ss, sleep=%ss)",
            self.mode,
            self.refresh_seconds,
            self.cycle_sleep_seconds,
        )

        next_refresh = datetime.now(timezone.utc)
        cycle_count = 0

        try:
            while True:
                cycle_count += 1
                now = datetime.now(timezone.utc)

                candles_by_symbol, finished = self.data_source.fetch_cycle()
                self.state_store.update_candles_batch(candles_by_symbol)

                pivot_updates, bot_updates = self.event_reader.read_updates()
                if pivot_updates:
                    self.state_store.ingest_pivot_zones(pivot_updates)
                if bot_updates:
                    self.state_store.ingest_bot_events(bot_updates)

                render_due = now >= next_refresh
                if self.mode == "development" and finished:
                    render_due = True

                if render_due:
                    ok = self.plot_builder.render(self.state_store.get_states())
                    if ok:
                        logger.info("Plot actualizado en %s", self.plot_builder.output_path)
                    next_refresh = now + timedelta(seconds=max(1, self.refresh_seconds))

                if self.mode == "development" and finished:
                    logger.info("Modo development finalizado: CSV agotado para todos los símbolos.")
                    break

                if self.max_cycles is not None and cycle_count >= self.max_cycles:
                    logger.info("Se alcanzó max_cycles=%s, finalizando runner.", self.max_cycles)
                    break

                if self.cycle_sleep_seconds > 0:
                    time.sleep(self.cycle_sleep_seconds)
        finally:
            self.data_source.close()


def _build_symbol_configs(mode: str) -> list[SymbolConfig]:
    cfg = load_settings(mode)
    symbols: list[SymbolConfig] = []
    for sym_cfg in cfg.symbols:
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


def build_runner(
    mode: str | None = None,
    refresh_seconds: int = 900,
    cycle_sleep_seconds: int | None = None,
    output_html: str = "outputs/visualizer_m3.html",
    pivot_log: str = "logs/pivot_zones.log",
    bot_events: str = "outputs/bot_events.jsonl",
    max_candles: int = 3000,
    max_events: int = 1500,
    max_cycles: int | None = None,
) -> VisualizerRunner:
    selected_mode = mode or ACTIVE_ENV
    cfg = load_settings(selected_mode)
    symbols = _build_symbol_configs(selected_mode)

    root = Path(__file__).resolve().parents[2]
    pivot_path = (root / pivot_log).resolve()
    events_path = _resolve_bot_events_path(root=root, configured_path=bot_events)
    output_path = (root / output_html).resolve()

    if cfg.data.data_mode == "development":
        data_source: BaseCandleDataSource = DevelopmentCandleDataSource(
            symbols=symbols,
            data_dir=(root / cfg.data.data_development_dir).resolve(),
            base_timeframe=cfg.data.csv_base_timeframe,
            lookback_days_entry=int(cfg.data.bootstrap_lookback_days_entry or cfg.data.bootstrap_lookback_days_zone),
            lookback_days_zone=int(cfg.data.bootstrap_lookback_days_zone),
            lookback_days_stop=int(cfg.data.bootstrap_lookback_days_stop or cfg.data.bootstrap_lookback_days_zone),
        )
        default_sleep = 0
    else:
        data_source = ProductionCandleDataSource(
            symbols=symbols,
            lookback_days_entry=int(cfg.data.bootstrap_lookback_days_entry or cfg.data.bootstrap_lookback_days_zone),
            lookback_days_zone=int(cfg.data.bootstrap_lookback_days_zone),
            lookback_days_stop=int(cfg.data.bootstrap_lookback_days_stop or cfg.data.bootstrap_lookback_days_zone),
            max_retries=cfg.broker.max_retries,
            retry_delay=cfg.broker.retry_delay,
            shared_data_getter=None,
        )
        default_sleep = 60

    event_reader = EventReader(
        pivot_log_path=pivot_path,
        bot_events_path=events_path,
        start_from_end=True,
    )
    state_store = VisualizerStateStore(
        symbols=[sym.name for sym in symbols],
        max_candles=max_candles,
        max_events=max_events,
    )
    plot_builder = BokehPlotBuilder(
        output_path=output_path,
        title=f"Visualizer M3 ({selected_mode}) - Pivot Zones + Entradas/Fills",
    )

    return VisualizerRunner(
        mode=selected_mode,
        data_source=data_source,
        event_reader=event_reader,
        state_store=state_store,
        plot_builder=plot_builder,
        refresh_seconds=int(refresh_seconds),
        cycle_sleep_seconds=default_sleep if cycle_sleep_seconds is None else int(cycle_sleep_seconds),
        max_cycles=max_cycles,
    )
