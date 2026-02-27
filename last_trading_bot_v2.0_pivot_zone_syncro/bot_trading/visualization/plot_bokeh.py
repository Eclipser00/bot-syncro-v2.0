"""Render del visualizador M3 usando directamente el plotting del backtest."""
from __future__ import annotations

import html as html_lib
import importlib.util
import logging
import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from .state_store import SymbolState

logger = logging.getLogger(__name__)

_ENTRY_EVENT_STYLE = (
    ("signal", "triangle", "#00c853"),
    ("order_fill", "circle", "#ff9800"),
    ("position", "diamond", "#40c4ff"),
)
_ZONE_COLORS = ["#ec407a", "#4caf50", "#3f51b5", "#f4c20d", "#ff8f00", "#26a69a"]
_RESPONSIVE_TAG = "data-live-fit-width"
_MAX_TRADE_LINES = 600


def safe_write_html(output_path: Path, html: str) -> bool:
    """Escribe HTML via archivo temporal + replace atomico.

    Si el archivo destino esta bloqueado (Windows), devuelve False sin lanzar.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(output_path.parent), suffix=".tmp") as tmp:
            tmp.write(html)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, output_path)
        return True
    except (PermissionError, OSError) as exc:
        logger.error(
            "No se pudo guardar el HTML '%s' (posible archivo abierto/bloqueado): %s",
            output_path,
            exc,
        )
        return False
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def _to_utc(ts: object) -> pd.Timestamp | None:
    try:
        value = pd.Timestamp(ts)
    except Exception:
        return None
    if value.tzinfo is None:
        return value.tz_localize("UTC")
    return value.tz_convert("UTC")


def _find_backtest_plotting_path() -> Path:
    """Ubica plotting.py del backtest bot."""
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    workspace_root = project_root.parent
    candidates = [
        workspace_root / "backtest bot v7.0 syncro" / "plotting.py",
        project_root / "plotting.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No se encontro plotting.py del backtest bot")


@lru_cache(maxsize=1)
def _load_backtest_plotter_class():
    """Carga dinamicamente Plot desde backtest bot plotting.py."""
    plotting_path = _find_backtest_plotting_path()
    spec = importlib.util.spec_from_file_location("backtest_plotting_bridge", plotting_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar spec para: {plotting_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plotter_class = getattr(module, "Plot", None)
    if plotter_class is None:
        raise RuntimeError(f"{plotting_path} no expone clase Plot")
    return plotter_class


def _sanitize_ohlcv(candles: pd.DataFrame) -> pd.DataFrame:
    if candles is None or candles.empty:
        return pd.DataFrame()
    frame = candles.copy().sort_index()
    idx = pd.DatetimeIndex(frame.index)
    if idx.tz is None:
        frame.index = idx.tz_localize("UTC")
    else:
        frame.index = idx.tz_convert("UTC")
    return frame


def _build_zone_indicators(candles: pd.DataFrame, zones: list[dict]) -> list[dict]:
    if candles.empty or not zones:
        return []

    idx = candles.index
    n = len(idx)
    visible = []
    for zone in zones[-24:]:
        ts = _to_utc(zone.get("ts_event"))
        if ts is None or ts > idx[-1]:
            continue
        visible.append((ts, zone))

    if not visible:
        return []

    indicators: list[dict] = []
    for zone_idx, (_, zone) in enumerate(visible[-8:]):
        start_ts = _to_utc(zone.get("ts_event"))
        if start_ts is None:
            continue
        start_pos = int(idx.searchsorted(start_ts, side="left"))
        if start_pos >= n:
            continue
        color = _ZONE_COLORS[zone_idx % len(_ZONE_COLORS)]

        top_values = np.full(n, np.nan)
        bot_values = np.full(n, np.nan)
        top_values[start_pos:] = float(zone.get("top", np.nan))
        bot_values[start_pos:] = float(zone.get("bot", np.nan))

        indicators.append(
            {
                "name": f"Zone_{zone_idx}_Top",
                "values": pd.Series(top_values),
                "overlay": True,
                "color": color,
                "line_style": "solid",
            }
        )
        indicators.append(
            {
                "name": f"Zone_{zone_idx}_Bot",
                "values": pd.Series(bot_values),
                "overlay": True,
                "color": color,
                "line_style": "solid",
            }
        )
    return indicators


def _build_entry_indicators(candles: pd.DataFrame, events: list[dict]) -> list[dict]:
    if candles.empty or not events:
        return []

    idx = candles.index
    n = len(idx)
    low = float(candles["low"].min())
    high = float(candles["high"].max())
    span = max(1e-9, high - low)
    lower = low - (5.0 * span)
    upper = high + (5.0 * span)

    indicators: list[dict] = []
    for event_type, marker, color in _ENTRY_EVENT_STYLE:
        values = np.full(n, np.nan)
        has_points = False

        for event in events:
            if str(event.get("event_type")) != event_type:
                continue

            ts = _to_utc(event.get("ts_m3"))
            if ts is None:
                continue
            if ts < idx[0] or ts > idx[-1]:
                continue

            price = float(event.get("price", 0.0))
            if not (price > 0.0 and lower <= price <= upper):
                continue

            bar_pos = int(idx.searchsorted(ts, side="left"))
            if bar_pos >= n:
                continue

            values[bar_pos] = price
            has_points = True

        if not has_points:
            continue

        indicators.append(
            {
                "name": f"Eventos_{event_type}",
                "values": pd.Series(values),
                "overlay": True,
                "marker": marker,
                "color": color,
                "line_style": "none",
            }
        )

    return indicators


def _build_trades_df(candles: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    """Construye trades cerrados (entry->exit) para que plotting.py dibuje lineas de posicion."""
    if candles.empty or not events:
        return pd.DataFrame(
            columns=["DatetimeOpen", "DatetimeClose", "Size", "EntryPrice", "ExitPrice", "PnL", "PnLComm"]
        )

    fill_events = []
    for event in events:
        if str(event.get("event_type")) != "order_fill":
            continue
        ts = _to_utc(event.get("ts_event"))
        if ts is None:
            continue
        price = float(event.get("price", 0.0) or 0.0)
        size = float(event.get("size", 0.0) or 0.0)
        if price <= 0.0 or size <= 0.0:
            continue
        side = str(event.get("side", "")).lower()
        reason = str(event.get("reason", "")).lower()
        fill_events.append(
            {
                "ts": ts,
                "price": price,
                "size": size,
                "side": side,
                "reason": reason,
                "bar_index": int(event.get("bar_index", -1)),
            }
        )

    if not fill_events:
        return pd.DataFrame(
            columns=["DatetimeOpen", "DatetimeClose", "Size", "EntryPrice", "ExitPrice", "PnL", "PnLComm"]
        )

    fill_events.sort(key=lambda e: (e["ts"], e["bar_index"]))

    open_by_side: dict[str, list[dict]] = {"long": [], "short": []}
    trades: list[dict] = []

    def _close_one(open_side: str, exit_evt: dict) -> None:
        stack = open_by_side.get(open_side)
        if not stack:
            return
        entry = stack.pop(0)
        direction = 1.0 if open_side == "long" else -1.0
        entry_price = float(entry["price"])
        exit_price = float(exit_evt["price"])
        qty = float(entry["size"])
        pnl = (exit_price - entry_price) * direction * qty
        trades.append(
            {
                "DatetimeOpen": entry["ts"],
                "DatetimeClose": exit_evt["ts"],
                "Size": qty * direction,
                "EntryPrice": entry_price,
                "ExitPrice": exit_price,
                "PnL": pnl,
                "PnLComm": pnl,
            }
        )

    for evt in fill_events:
        side = evt["side"]
        reason = evt["reason"]

        if reason.startswith("close_"):
            if side == "short":
                _close_one("long", evt)
            elif side == "long":
                _close_one("short", evt)
            elif side == "flat":
                if open_by_side["long"]:
                    _close_one("long", evt)
                elif open_by_side["short"]:
                    _close_one("short", evt)
            continue

        if side in {"long", "short"}:
            open_by_side[side].append(evt)
            continue

        if side == "flat":
            if open_by_side["long"]:
                _close_one("long", evt)
            elif open_by_side["short"]:
                _close_one("short", evt)

    if not trades:
        return pd.DataFrame(
            columns=["DatetimeOpen", "DatetimeClose", "Size", "EntryPrice", "ExitPrice", "PnL", "PnLComm"]
        )

    out = pd.DataFrame(trades).sort_values("DatetimeClose").tail(_MAX_TRADE_LINES).reset_index(drop=True)
    return out


def _render_single_symbol_html(symbol: str, state: SymbolState, target_dir: Path) -> str:
    candles = _sanitize_ohlcv(state.candles_m3)
    if candles.empty:
        return f"<html><body><h2>{symbol} - M3</h2><p>Sin datos para graficar.</p></body></html>"

    indicators = []
    indicators.extend(_build_zone_indicators(candles, state.saved_zones))
    indicators.extend(_build_entry_indicators(candles, state.entry_events))
    trades_df = _build_trades_df(candles, state.entry_events)

    plotter_class = _load_backtest_plotter_class()
    plotter = plotter_class()

    tmp_html = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(target_dir), suffix=f"_{symbol}.html") as tmp:
            tmp_html = Path(tmp.name)
        with warnings.catch_warnings():
            # Warning benigno de Bokeh al fusionar toolbars en gridplot del plotting heredado.
            warnings.filterwarnings(
                "ignore",
                message="found multiple competing values for 'toolbar.active_drag' property; using the latest value",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="found multiple competing values for 'toolbar.active_scroll' property; using the latest value",
                category=UserWarning,
            )
            plotter.max_min_plot(
                ohlcv=candles,
                equity_curve=None,
                trades_df=trades_df,
                indicators=indicators,
                filename=str(tmp_html),
            )
        raw_html = tmp_html.read_text(encoding="utf-8")
        return _inject_responsive_fit(raw_html)
    finally:
        if tmp_html is not None and tmp_html.exists():
            try:
                tmp_html.unlink(missing_ok=True)
            except Exception:
                pass


def _compose_multi_symbol_html(title: str, symbol_html: dict[str, str]) -> str:
    if not symbol_html:
        return "<html><body><h2>Visualizer M3</h2><p>Sin datos para graficar.</p></body></html>"
    if len(symbol_html) == 1:
        return next(iter(symbol_html.values()))

    sections = []
    for symbol, html in symbol_html.items():
        escaped = html_lib.escape(html, quote=True)
        sections.append(
            "<section>"
            f"<h2>{symbol} - M3</h2>"
            f"<iframe srcdoc=\"{escaped}\" loading=\"lazy\"></iframe>"
            "</section>"
        )
    return (
        "<html><head><meta charset=\"utf-8\">"
        f"<title>{title}</title>"
        "<style>"
        "body{margin:16px;font-family:Arial,sans-serif;background:#111;color:#eee;}"
        "section{margin-bottom:24px;}"
        "h2{font-size:18px;margin:0 0 8px 0;}"
        "iframe{width:100%;height:980px;border:1px solid #333;background:#fff;}"
        "</style></head><body>"
        + "".join(sections)
        + "</body></html>"
    )


def _inject_responsive_fit(html: str) -> str:
    """Inyecta un ajuste de escala para que el plot no desborde pantallas pequenas."""
    if _RESPONSIVE_TAG in html:
        return html

    script = (
        f"<script {_RESPONSIVE_TAG}=\"1\">"
        "(function(){"
        "function fitAll(){"
        "var roots=document.querySelectorAll('.bk-root');"
        "if(!roots||!roots.length){return;}"
        "var viewport=Math.max(320,(window.innerWidth||document.documentElement.clientWidth||1850)-24);"
        "for(var i=0;i<roots.length;i++){"
        "var root=roots[i];"
        "root.style.transform='none';"
        "root.style.transformOrigin='top left';"
        "var baseWidth=root.getBoundingClientRect().width||root.scrollWidth||0;"
        "var baseHeight=root.getBoundingClientRect().height||root.scrollHeight||0;"
        "if(!baseWidth||!baseHeight){continue;}"
        "var scale=Math.min(1,viewport/baseWidth);"
        "root.style.transform='scale('+scale.toFixed(4)+')';"
        "var holder=root.parentElement;"
        "if(holder){"
        "holder.style.height=Math.ceil(baseHeight*scale+8)+'px';"
        "holder.style.overflow='visible';"
        "}"
        "}"
        "}"
        "window.addEventListener('resize',fitAll);"
        "window.addEventListener('load',function(){"
        "var tries=0;"
        "var timer=setInterval(function(){fitAll();tries+=1;if(tries>25){clearInterval(timer);}},120);"
        "});"
        "})();"
        "</script>"
    )

    if "</body>" in html:
        return html.replace("</body>", script + "</body>", 1)
    return html + script


@dataclass
class BokehPlotBuilder:
    """Construye y guarda el HTML del visualizador M3."""

    output_path: Path
    title: str = "Visualizer M3 - Pivot Zones"

    def build_html(self, states: dict[str, SymbolState]) -> str:
        html_by_symbol: dict[str, str] = {}
        for symbol, state in states.items():
            if state.candles_m3.empty:
                continue
            html_by_symbol[symbol] = _render_single_symbol_html(symbol, state, self.output_path.parent)
        return _compose_multi_symbol_html(self.title, html_by_symbol)

    def render(self, states: dict[str, SymbolState]) -> bool:
        try:
            html = self.build_html(states)
        except Exception as exc:
            logger.error("Fallo construyendo HTML del visualizador: %s", exc, exc_info=True)
            return False
        return safe_write_html(self.output_path, html)
