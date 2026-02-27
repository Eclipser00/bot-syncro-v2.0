from __future__ import annotations

from datetime import timezone
import os

import pandas as pd

from bot_trading.domain.entities import SymbolConfig
from bot_trading.visualization.event_reader import EventReader
from bot_trading.visualization.live_plot_service import AutoVisualizerService
from bot_trading.visualization.plot_bokeh import BokehPlotBuilder, _build_entry_indicators, _build_trades_df, safe_write_html
from bot_trading.visualization.runner import _resolve_bot_events_path
from bot_trading.visualization.state_store import SymbolState, VisualizerStateStore


def test_event_reader_incremental_offsets(tmp_path) -> None:
    pivot_log = tmp_path / "pivot_zones.log"
    events_log = tmp_path / "bot_events.jsonl"

    pivot_log.write_text(
        "time=2026-02-11T00:57:00+00:00 symbol=EURUSD tf=M3 top=1.2 bot=1.1 mid=1.15 width=0.05 pivots=5 bar=85 total_saved=1\n",
        encoding="utf-8",
    )
    events_log.write_text(
        '{"ts_event":"2026-02-11T00:58:00+00:00","symbol":"OANDA_EURUSD, 1_x","event_type":"signal","side":"long","price":1.16,"size":0.1}\n',
        encoding="utf-8",
    )

    reader = EventReader(pivot_log_path=pivot_log, bot_events_path=events_log)
    pivots_1, events_1 = reader.read_updates()
    assert len(pivots_1) == 1
    assert len(events_1) == 1
    assert pivots_1[0]["symbol"] == "EURUSD"
    assert events_1[0]["symbol"] == "EURUSD"

    # Sin líneas nuevas, no debe reprocesar histórico.
    pivots_2, events_2 = reader.read_updates()
    assert pivots_2 == []
    assert events_2 == []

    with events_log.open("a", encoding="utf-8") as fh:
        fh.write(
            '{"ts_event":"2026-02-11T01:00:00+00:00","symbol":"GBPUSD","event_type":"order_fill","side":"short","price":1.30,"size":0.2}\n'
        )

    _, events_3 = reader.read_updates()
    assert len(events_3) == 1
    assert events_3[0]["symbol"] == "GBPUSD"
    assert events_3[0]["event_type"] == "order_fill"


def test_state_store_projecta_eventos_a_m3() -> None:
    idx = pd.date_range("2026-02-11 00:00:00+00:00", periods=4, freq="3min")
    candles = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.1, 1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1, 1.2],
            "close": [1.05, 1.15, 1.25, 1.35],
            "volume": [10, 11, 12, 13],
        },
        index=idx,
    )

    store = VisualizerStateStore(symbols=["EURUSD"], max_candles=100, max_events=100)
    store.update_candles("EURUSD", candles)
    store.ingest_pivot_zones(
        [
            {
                "symbol": "EURUSD",
                "ts_event": "2026-02-11T00:03:00+00:00",
                "top": 1.20,
                "bot": 1.10,
                "mid": 1.15,
                "width": 0.05,
                "bar": 10,
                "pivots": 3,
                "total_saved": 1,
            }
        ]
    )
    store.ingest_bot_events(
        [
            {
                "symbol": "EURUSD",
                "ts_event": "2026-02-11T00:04:10+00:00",
                "event_type": "signal",
                "side": "long",
                "price": 1.16,
                "size": 0.3,
                "reason": "breakout_long",
                "order_id": "abc",
                "bar_index": 11,
                "timeframe": "M1",
            }
        ]
    )

    state = store.get_symbol_state("EURUSD")
    assert state is not None
    assert len(state.candles_m3) == 4
    assert len(state.saved_zones) == 1
    assert len(state.entry_events) == 1
    # 00:04:10 se proyecta al cierre de vela M3: 00:06:00.
    assert state.entry_events[0]["ts_m3"] == pd.Timestamp("2026-02-11T00:06:00+00:00")
    assert state.entry_events[0]["ts_m3"].tz_convert(timezone.utc).minute == 6


def test_safe_write_html_no_crashea_si_archivo_bloqueado(tmp_path, monkeypatch) -> None:
    output = tmp_path / "plot.html"

    def _raise_permission(_src, _dst):
        raise PermissionError("file locked")

    monkeypatch.setattr(os, "replace", _raise_permission)
    ok = safe_write_html(output, "<html><body>test</body></html>")
    assert ok is False


def test_resolve_bot_events_path_prioriza_mas_reciente_en_ruta_padre(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    parent_root = tmp_path
    (repo_root / "outputs").mkdir(parents=True)
    (parent_root / "outputs").mkdir(parents=True, exist_ok=True)

    local_events = repo_root / "outputs" / "bot_events.jsonl"
    parent_events = parent_root / "outputs" / "bot_events.jsonl"
    local_events.write_text('{"event_type":"local"}\n', encoding="utf-8")
    parent_events.write_text('{"event_type":"parent"}\n', encoding="utf-8")

    # Forzamos que parent sea más reciente.
    os.utime(local_events, (1, 1))
    os.utime(parent_events, (2, 2))

    resolved = _resolve_bot_events_path(repo_root, "outputs/bot_events.jsonl")
    assert resolved == parent_events.resolve()


def test_event_reader_start_from_end_omite_historico_inicial(tmp_path) -> None:
    pivot_log = tmp_path / "pivot_zones.log"
    events_log = tmp_path / "bot_events.jsonl"
    pivot_log.write_text("time=2026-02-11T00:57:00+00:00 symbol=EURUSD tf=M3 top=1.2 bot=1.1 mid=1.15 width=0.05 pivots=5 bar=85 total_saved=1\n", encoding="utf-8")
    events_log.write_text('{"ts_event":"2026-02-11T00:58:00+00:00","symbol":"EURUSD","event_type":"signal","side":"long","price":1.16,"size":0.1}\n', encoding="utf-8")

    reader = EventReader(pivot_log_path=pivot_log, bot_events_path=events_log, start_from_end=True)
    pivots_1, events_1 = reader.read_updates()
    assert pivots_1 == []
    assert events_1 == []

    with pivot_log.open("a", encoding="utf-8") as fh:
        fh.write("time=2026-02-11T01:00:00+00:00 symbol=EURUSD tf=M3 top=1.21 bot=1.11 mid=1.16 width=0.05 pivots=5 bar=86 total_saved=2\n")
    with events_log.open("a", encoding="utf-8") as fh:
        fh.write('{"ts_event":"2026-02-11T01:01:00+00:00","symbol":"EURUSD","event_type":"order_fill","side":"long","price":1.17,"size":0.1}\n')

    pivots_2, events_2 = reader.read_updates()
    assert len(pivots_2) == 1
    assert len(events_2) == 1


def test_plot_builder_filtra_eventos_fuera_de_rango(tmp_path) -> None:
    idx = pd.date_range("2026-02-11 00:00:00+00:00", periods=4, freq="3min")
    candles = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.1, 1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1, 1.2],
            "close": [1.05, 1.15, 1.25, 1.35],
            "volume": [10, 11, 12, 13],
        },
        index=idx,
    )

    state = SymbolState(candles_m3=candles)
    state.entry_events = [
        {
            "ts_event": pd.Timestamp("2026-02-11T00:04:10+00:00"),
            "ts_m3": pd.Timestamp("2026-02-11T00:06:00+00:00"),
            "event_type": "signal",
            "price": 1.16,
            "size": 0.1,
            "side": "long",
            "reason": "ok",
            "order_id": "1",
            "bar_index": 1,
            "timeframe": "M1",
        },
        {
            "ts_event": pd.Timestamp("2026-02-11T00:04:10+00:00"),
            "ts_m3": pd.Timestamp("2026-02-11T00:06:00+00:00"),
            "event_type": "signal",
            "price": 100.0,  # outlier que antes reventaba el eje.
            "size": 0.1,
            "side": "long",
            "reason": "bad",
            "order_id": "2",
            "bar_index": 1,
            "timeframe": "M1",
        },
    ]

    indicators = _build_entry_indicators(candles, state.entry_events)
    signal_series = [ind["values"] for ind in indicators if ind["name"] == "Eventos_signal"][0]
    valid_prices = [float(v) for v in signal_series.dropna().tolist()]

    # Debe conservar solo el evento valido y descartar el outlier.
    assert valid_prices == [1.16]


def test_auto_visualizer_genera_html_por_ticker(tmp_path) -> None:
    symbols = [
        SymbolConfig(name="EURUSD", min_timeframe="M1"),
        SymbolConfig(name="GBPUSD", min_timeframe="M1"),
    ]
    pivot_log = tmp_path / "pivot_zones.log"
    events_log = tmp_path / "bot_events.jsonl"
    pivot_log.write_text("", encoding="utf-8")
    events_log.write_text("", encoding="utf-8")

    service = AutoVisualizerService(
        symbols=symbols,
        output_dir=tmp_path,
        pivot_log_path=pivot_log,
        bot_events_path=events_log,
        refresh_seconds=1,
        start_from_end=True,
    )

    idx = pd.date_range("2026-02-11 00:00:00+00:00", periods=3, freq="3min")
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.01, 1.02],
            "high": [1.02, 1.03, 1.04],
            "low": [0.99, 1.00, 1.01],
            "close": [1.01, 1.02, 1.03],
            "volume": [1.0, 1.0, 1.0],
        },
        index=idx,
    )

    now = pd.Timestamp("2026-02-11T00:06:00+00:00").to_pydatetime()
    service.on_market_data(symbols[0], {"M3": frame}, now)
    service.on_market_data(symbols[1], {"M3": frame}, now)
    service.close()

    assert (tmp_path / "visualizerEURUSD.html").exists()
    assert (tmp_path / "visualizerGBPUSD.html").exists()


def test_plot_builder_inyecta_autofit_para_pantalla(tmp_path) -> None:
    idx = pd.date_range("2026-02-11 00:00:00+00:00", periods=3, freq="3min")
    candles = pd.DataFrame(
        {
            "open": [1.0, 1.01, 1.02],
            "high": [1.02, 1.03, 1.04],
            "low": [0.99, 1.00, 1.01],
            "close": [1.01, 1.02, 1.03],
            "volume": [1.0, 1.0, 1.0],
        },
        index=idx,
    )
    state = SymbolState(candles_m3=candles)
    builder = BokehPlotBuilder(output_path=tmp_path / "plot.html")
    html = builder.build_html({"EURUSD": state})
    assert 'data-live-fit-width="1"' in html


def test_build_trades_df_desde_order_fill_open_close() -> None:
    idx = pd.date_range("2026-02-11 00:00:00+00:00", periods=8, freq="3min")
    candles = pd.DataFrame(
        {
            "open": [1.10, 1.11, 1.12, 1.13, 1.12, 1.11, 1.10, 1.09],
            "high": [1.11, 1.12, 1.13, 1.14, 1.13, 1.12, 1.11, 1.10],
            "low": [1.09, 1.10, 1.11, 1.12, 1.11, 1.10, 1.09, 1.08],
            "close": [1.105, 1.115, 1.125, 1.135, 1.115, 1.105, 1.095, 1.085],
            "volume": [1.0] * 8,
        },
        index=idx,
    )
    events = [
        {
            "event_type": "order_fill",
            "ts_event": "2026-02-11T00:06:00+00:00",
            "side": "long",
            "price": 1.1250,
            "size": 0.20,
            "reason": "fill",
            "bar_index": 2,
        },
        {
            "event_type": "order_fill",
            "ts_event": "2026-02-11T00:15:00+00:00",
            "side": "short",
            "price": 1.1050,
            "size": 0.20,
            "reason": "close_stop",
            "bar_index": 5,
        },
    ]

    trades_df = _build_trades_df(candles, events)
    assert len(trades_df) == 1
    trade = trades_df.iloc[0]
    assert trade["Size"] > 0
    assert trade["EntryPrice"] == 1.1250
    assert trade["ExitPrice"] == 1.1050
    assert trade["PnL"] < 0
