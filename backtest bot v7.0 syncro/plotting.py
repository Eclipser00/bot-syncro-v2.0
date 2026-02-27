# plotting.py
# ================================================================
# Sistema de plotting interactivo con Bokeh (refactor a clases)
# - Clase Plot.max_min_plot(...) -> integra tu antigua función plot(...)
# - Clase Heatmaps.generate(...) -> integra tu antigua plot_heatmaps(...)
# - Soporta columnas lower-case (open/high/low/close/volume) o TitleCase
# - Mapea automáticamente trades_df desde tu Backtest.run_backtrader_core:
#     open_datetime/close_datetime/size/price_open/price_close/pnl/pnl_comm
#   y también soporta tu esquema anterior (DatetimeOpen/DatetimeClose/...).
# - Logs MUY verbosos y comentarios extensos para depuración.
# ================================================================

from __future__ import annotations

import os
from itertools import cycle, combinations
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

# ----------------- Bokeh -----------------
from bokeh.colors import RGB
from bokeh.colors.named import lime as BULL_COLOR, tomato as BEAR_COLOR
from bokeh.layouts import gridplot
from bokeh.models import (
    CrosshairTool, CustomJS, ColumnDataSource, CustomJSTransform,
    NumeralTickFormatter, Span, HoverTool, Range1d,
    DatetimeTickFormatter, WheelZoomTool, LinearColorMapper,
    BasicTicker, ColorBar
)
try:
    # Bokeh >=3
    from bokeh.models import CustomJSTickFormatter
except ImportError:
    # Bokeh <3 (fallback)
    from bokeh.models import FuncTickFormatter as CustomJSTickFormatter

from bokeh.palettes import Category10, Viridis256
from bokeh.plotting import figure as _figure, save, output_file
from bokeh.transform import factor_cmap, transform

# ----------------- JS autoscale opcional -----------------
_AUTOSCALE_JS_CALLBACK = ""
try:
    with open(os.path.join(os.path.dirname(__file__), 'autoscale_cb.js'), encoding='utf-8') as _f:
        _AUTOSCALE_JS_CALLBACK = _f.read()
    print("[PLOTTING] autoscale_cb.js cargado OK")
except Exception as e:
    print(f"[PLOTTING][WARN] autoscale_cb.js no disponible ({e}). Se desactiva autoscale JS.")

# ===================== CONSTANTES =========================
_MAX_CANDLES = 30000          # seguridad por si nos pasan series gigantes
_INDICATOR_HEIGHT = 50          # altura por panel de indicador/oscillator
NBSP = '\N{NBSP}' * 4           # espacios duros para tooltips

print("[PLOTTING] Modulo plotting.py cargado correctamente")


# =========================================================
# =============== HELPERS/UTILIDADES INTERNAS =============
# =========================================================

def _log(msg: str) -> None:
    """Logger ultra-verboso (puedes comentar los prints si molestan)."""
    print(msg)
# Comentario:
# Centralizamos el logging para poder silenciar fácilmente todos los prints, si fuera necesario.


def _colorgen():
    """Generador infinito de colores distinguibles (Category10)."""
    yield from cycle(Category10[10])
# Comentario:
# Evitamos colores repetidos al ir solicitando colores para múltiples indicadores.


def _lightness(color: RGB, lightness_val: float = .94) -> RGB:
    """
    Ajusta el brillo de un color RGB de Bokeh.
    lightness_val: 0.0 (negro) a 1.0 (blanco)
    """
    # Convertimos vía HLS para subir/bajar luminosidad
    from colorsys import hls_to_rgb, rgb_to_hls
    rgb = np.array([color.r, color.g, color.b]) / 255
    h, _, s = rgb_to_hls(*rgb)
    rgb = (np.array(hls_to_rgb(h, lightness_val, s)) * 255).astype(int)
    return RGB(*rgb)
# Comentario:
# Esta función se usa para oscurecer/clarificar colores base (p.ej., trades ganadores/perdedores).


def _normalize_ohlcv_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas OHLCV a TitleCase como espera el plot:
        Open, High, Low, Close, Volume
    Acepta variantes lower-case: open/high/low/close/volume
    """
    mapping = {
        'open': 'Open', 'Open': 'Open',
        'high': 'High', 'High': 'High',
        'low': 'Low', 'Low': 'Low',
        'close': 'Close', 'Close': 'Close',
        'volume': 'Volume', 'Volume': 'Volume',
    }
    cols = {c: mapping.get(c, c) for c in df.columns}
    out = df.rename(columns=cols).copy()
    needed = {'Open', 'High', 'Low', 'Close'}
    if not needed.issubset(out.columns):
        raise ValueError(f"[PLOTTING] Faltan columnas OHLC en DataFrame: {out.columns}")
    if 'Volume' not in out.columns:
        out['Volume'] = 0.0
    return out
# Comentario:
# Nuestro DataManager entrega columnas en minúscula; tu plot anterior usaba TitleCase.
# Aquí unificamos para no romper nada.


def _normalize_trades_cols(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza trades a las columnas que necesita el plot:
      DatetimeOpen, DatetimeClose, Size, EntryPrice, ExitPrice, PnL, PnLComm
    Acepta tanto tu nuevo formato (open_datetime, price_open, ...) como el antiguo.
    """
    if trades_df is None or len(trades_df) == 0:
        return pd.DataFrame(columns=[
            'DatetimeOpen', 'DatetimeClose', 'Size', 'EntryPrice', 'ExitPrice', 'PnL', 'PnLComm'
        ])

    df = trades_df.copy()

    # Mapping desde tu Backtest._TradeList (nuevo esquema)
    map_new = {
        'open_datetime': 'DatetimeOpen',
        'close_datetime': 'DatetimeClose',
        'size': 'Size',
        'price_open': 'EntryPrice',
        'price_close': 'ExitPrice',
        'pnl': 'PnL',
        'pnl_comm': 'PnLComm'
    }

    # Mapping del esquema anterior (si ya viene con los nombres viejos, se mantienen)
    map_old = {
        'DatetimeOpen': 'DatetimeOpen',
        'DatetimeClose': 'DatetimeClose',
        'Size': 'Size',
        'EntryPrice': 'EntryPrice',
        'ExitPrice': 'ExitPrice',
        'PnL': 'PnL',
        'PnLComm': 'PnLComm'
    }

    # Elegimos qué mapping aplicar
    if set(map_new.keys()).issubset(df.columns):
        df = df.rename(columns=map_new)
    elif set(map_old.keys()).issubset(df.columns):
        # Ya viene en el formato esperado; nada que hacer
        pass
    else:
        # Intento heurístico mínimo: lowercase -> TitleCase
        lower = {c.lower(): c for c in df.columns}
        if 'open_datetime' in lower and 'close_datetime' in lower:
            df = df.rename(columns={lower.get('open_datetime'): 'DatetimeOpen',
                                    lower.get('close_datetime'): 'DatetimeClose'})
        if 'size' in lower: df = df.rename(columns={lower.get('size'): 'Size'})
        if 'price_open' in lower: df = df.rename(columns={lower.get('price_open'): 'EntryPrice'})
        if 'price_close' in lower: df = df.rename(columns={lower.get('price_close'): 'ExitPrice'})
        if 'pnl' in lower: df = df.rename(columns={lower.get('pnl'): 'PnL'})
        if 'pnl_comm' in lower: df = df.rename(columns={lower.get('pnl_comm'): 'PnLComm'})

    # Asegurar tipos mínimos
    for col in ['DatetimeOpen', 'DatetimeClose']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    for col in ['Size', 'EntryPrice', 'ExitPrice', 'PnL', 'PnLComm']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Orden de columnas final
    cols = ['DatetimeOpen', 'DatetimeClose', 'Size', 'EntryPrice', 'ExitPrice', 'PnL', 'PnLComm']
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]
# Comentario:
# Esto te permite plotear directamente el DataFrame de trades que devuelve tu run_backtrader_core,
# o el de tu versión anterior sin cambiar nada en otros módulos.


# =========================================================
# ====================== CLASE PLOT =======================
# =========================================================
class Plot:
    def max_min_plot(
        self,
        ohlcv: pd.DataFrame,
        equity_curve: Optional[pd.Series] = None,
        trades_df: Optional[pd.DataFrame] = None,
        indicators: Optional[List[Dict[str, Any]]] = None,
        filename: str = ""
    ):
        """
        Crea un HTML interactivo con:
          - Panel de Equity + Drawdown (si equity_curve proporcionado)
          - Velas OHLC con overlay de indicadores y líneas de trades
          - Panel de P/L por trade (si trades_df)
          - Panel de Volumen (si existe columna Volume)

        Parámetros:
          ohlcv: DataFrame con columnas open/high/low/close/volume (soportado) o TitleCase.
          equity_curve: Serie de equity (opcional).
          trades_df: DataFrame de trades (puede venir de Backtest.run_backtrader_core).
          indicators: Lista opcional de dicts con:
              {'name': str, 'values': pd.Series, 'overlay': bool,
               'marker': 'circle'|'triangle'|..., 'color': '#rrggbb'|'named', 'line_style': 'solid'|'none'}
          filename: ruta de salida .html

        Retorna:
          fig (layout de Bokeh) para inspección o tests.
        """
        # ================== PREPARACIÓN Y LOGS ==================
        _log("\n[PLOT] Iniciando plot interactivo...")
        if indicators is None:
            indicators = []

        # 1) Normalizar OHLCV y recortar tamaño si es enorme
        df = _normalize_ohlcv_cols(ohlcv)
        if len(df) > _MAX_CANDLES:
            _log(f"[PLOT][WARN] Serie > {_MAX_CANDLES} velas. Se recorta para seguridad.")
            df = df.iloc[-_MAX_CANDLES:].copy()

        df.index.name = None
        df['datetime'] = df.index
        df = df.reset_index(drop=True)
        index = df.index

        # IMPORTANTE: Sincronizar indicadores con el tamaño recortado del OHLC
        if indicators:
            _log(f"[PLOT] Sincronizando {len(indicators)} indicadores con OHLC recortado...")
            for ind_info in indicators:
                if 'values' in ind_info and hasattr(ind_info['values'], 'iloc'):
                    original_len = len(ind_info['values'])
                    # Recortar el indicador al mismo tamaño que el OHLC recortado
                    ind_info['values'] = ind_info['values'].iloc[-len(df):].reset_index(drop=True)
                    _log(f"[PLOT] Indicador {ind_info.get('name', 'unknown')}: {original_len} -> {len(ind_info['values'])} valores")

        # 2) Normalizar trades (soporta ambos formatos)
        trades = _normalize_trades_cols(trades_df) if trades_df is not None else pd.DataFrame()

        # IMPORTANTE: Sincronizar equity_curve con el tamaño recortado del OHLC
        if equity_curve is not None and len(equity_curve) > 0:
            original_equity_len = len(equity_curve)
            equity_curve = equity_curve.iloc[-len(df):].reset_index(drop=True)
            _log(f"[PLOT] Equity sincronizado: {original_equity_len} -> {len(equity_curve)} valores")

        # 3) Globos de control: dibujar equity/drawdown/volumen/trades si procede
        plot_equity = equity_curve is not None and len(equity_curve) > 0
        plot_drawdown = plot_equity
        plot_volume = 'Volume' in df.columns and not df['Volume'].isna().all()
        plot_trades = trades is not None and len(trades) > 0
        plot_pl = plot_trades

        _log(f"[PLOT] OHLCV: {len(ohlcv)} velas  |  Equity: {len(equity_curve) if equity_curve is not None else 0} puntos  |  Trades: {len(trades)}")

        # 4) Salida
        if not filename:
            filename = 'backtest_plot.html'
        if not filename.endswith('.html'):
            filename += '.html'
        output_file(filename, title=filename)

        # ================== FIGURAS BASE ==================
        # Eje X por índice (lo hacemos lineal para mayor control; formateamos labels con datetime)
        is_datetime_index = isinstance(ohlcv.index, pd.DatetimeIndex)
        pad = (index[-1] - index[0]) / 20 if len(index) > 1 else 1

        new_bokeh_figure = lambda **kw: _figure(
            x_axis_type='linear',
            width=1850,
            tools="xpan,xwheel_zoom,box_zoom,undo,redo,reset,save",
            active_drag='xpan',
            active_scroll='xwheel_zoom',
            **kw
        )

        _kwargs = dict(
            x_range=Range1d(index[0], index[-1], min_interval=10, bounds=(index[0] - pad, index[-1] + pad))
        ) if len(index) > 1 else {}

        fig_ohlc = new_bokeh_figure(height=400, **_kwargs)
        figs_above_ohlc, figs_below_ohlc = [], []

        # Fuente principal
        source = ColumnDataSource(df)
        source.add((df['Close'] >= df['Open']).values.astype(np.uint8).astype(str), 'inc')

        # Colores y formato de velas
        COLORS = [BEAR_COLOR, BULL_COLOR]
        BAR_WIDTH = .8
        inc_cmap = factor_cmap('inc', COLORS, ['0', '1'])

        # Formateo de ejes con datetime (tick formatter JS)
        if is_datetime_index:
            fig_ohlc.xaxis.formatter = CustomJSTickFormatter(
                args=dict(axis=fig_ohlc.xaxis[0],
                          formatter=DatetimeTickFormatter(days='%a, %d %b', months='%m/%Y'),
                          source=source),
                code='''
this.labels = this.labels || formatter.doFormat(ticks
                                                .map(i => source.data.datetime[i])
                                                .filter(t => t !== undefined));
return this.labels[index] || "";
                '''
            )

        def new_indicator_figure(**kwargs):
            kwargs.setdefault('height', _INDICATOR_HEIGHT)
            fig = new_bokeh_figure(x_range=fig_ohlc.x_range, **kwargs)
            fig.xaxis.visible = False
            fig.yaxis.minor_tick_line_color = None
            fig.yaxis.ticker.desired_num_ticks = 3
            return fig

        def set_tooltips(fig, tooltips=(), vline=True, renderers=()):
            tooltips = list(tooltips)
            renderers = list(renderers)
            if is_datetime_index:
                formatters = {'@datetime': 'datetime'}
                tooltips = [("Date", "@datetime{%c}")] + tooltips
            else:
                formatters = {}
                tooltips = [("#", "@index")] + tooltips
            fig.add_tools(HoverTool(
                point_policy='follow_mouse',
                renderers=renderers,
                formatters=formatters,
                tooltips=tooltips,
                mode='vline' if vline else 'mouse'
            ))

        _log("[PLOT] Figuras base creadas OK")

        # ================== PANEL DE EQUITY ==================
        if plot_equity:
            _log("[PLOT] Renderizando panel de Equity...")
            fig_equity = new_indicator_figure(y_axis_label="Equity", height=100)

            # Alinear equity a índice de OHLC (usar el índice numérico de df, no ohlcv.index)
            eq = equity_curve  # Ya está sincronizado con df.index
            source.add(eq, 'equity')

            r_equity = fig_equity.line('index', 'equity', source=source, line_width=1.5, line_color='blue')

            # Marcadores: Peak y Final
            peak_idx = int(eq.values.argmax())
            peak_val = float(eq.iloc[peak_idx])
            final_val = float(eq.iloc[-1])

            fig_equity.scatter(peak_idx, peak_val, legend_label=f'Peak (${peak_val:,.0f})', color='cyan', size=8)
            fig_equity.scatter(index[-1], final_val, legend_label=f'Final (${final_val:,.0f})', color='blue', size=8)

            set_tooltips(fig_equity, [("Equity", "@equity{$ 0,0}")], renderers=[r_equity])
            fig_equity.yaxis.formatter = NumeralTickFormatter(format="$ 0.0 a")
            figs_above_ohlc.append(fig_equity)

        # ================== PANEL DE DRAWDOWN ==================
        if plot_drawdown and plot_equity:
            _log("[PLOT] Renderizando panel de Drawdown...")
            fig_dd = new_indicator_figure(y_axis_label="Drawdown", height=80)

            eq_data = source.data['equity']
            running_max = pd.Series(eq_data).cummax()
            drawdown = (pd.Series(eq_data) - running_max) / running_max
            source.add(drawdown.values, 'drawdown')

            r_dd = fig_dd.line('index', 'drawdown', source=source, line_width=1.3, line_color='red')

            max_dd_idx = int(np.nanargmin(drawdown.values))
            max_dd_val = float(drawdown.iloc[max_dd_idx])
            fig_dd.scatter(max_dd_idx, max_dd_val, legend_label=f'Peak (-{abs(max_dd_val)*100:.1f}%)', color='red', size=8)

            set_tooltips(fig_dd, [("Drawdown", "@drawdown{-0.[0]%}")], renderers=[r_dd])
            fig_dd.yaxis.formatter = NumeralTickFormatter(format="-0.[0]%")
            figs_above_ohlc.append(fig_dd)

        _log("[PLOT] Paneles superiores OK")

        # ================== PANEL OHLC PRINCIPAL ==================
        _log("[PLOT] Renderizando velas OHLC...")
        # Segmentos High-Low y velas Open-Close
        fig_ohlc.segment('index', 'High', 'index', 'Low', source=source, color="black", legend_label='OHLC')
        r_ohlc = fig_ohlc.vbar('index', BAR_WIDTH, 'Open', 'Close', source=source,
                               line_color="black", fill_color=inc_cmap, legend_label='OHLC')

        # Preparamos series para autoscale (mínimos/máximos por fila)
        ohlc_extreme_values = df[['High', 'Low']].copy()

        # Tooltips base del panel OHLC
        ohlc_tooltips = [
            ('x, y', NBSP.join(('$index', '$y{0,0.0[0000]}'))),
            ('OHLC', NBSP.join(('@Open{0,0.0[0000]}', '@High{0,0.0[0000]}',
                                '@Low{0,0.0[0000]}', '@Close{0,0.0[0000]}'))),
            ('Volume', '@Volume{0,0}')
        ]

        # ================== TRADES SOBRE OHLC ==================
        if plot_trades:
            _log(f"[PLOT] Renderizando {len(trades)} trades...")

            # Mapeo datetime -> índice (para dibujar en x por posición)
            def _closest_idx(dt: pd.Timestamp) -> int:
                # Buscar la fecha más cercana en df['datetime']
                if dt in df['datetime'].values:
                    return int(df[df['datetime'] == dt].index[0])
                
                # Si no está exacta, buscar la más cercana
                time_diffs = np.abs((df['datetime'] - dt).dt.total_seconds())
                closest_idx = time_diffs.idxmin()
                return int(closest_idx)

            entry_bars = [ _closest_idx(dt) for dt in trades['DatetimeOpen'] ]
            exit_bars  = [ _closest_idx(dt) for dt in trades['DatetimeClose'] ]

            # Retorno por trade aproximado (PnL neto sobre valor noz)
            with np.errstate(divide='ignore', invalid='ignore'):
                noz = (np.abs(trades['Size']) * trades['EntryPrice']).replace(0, np.nan)
                returns_pct = trades['PnLComm'] / noz

            returns_positive = (returns_pct > 0).astype(int).astype(str)

            trade_source = ColumnDataSource(dict(
                entry_bar=entry_bars,
                exit_bar=exit_bars,
                entry_datetime=trades['DatetimeOpen'].values,
                exit_datetime=trades['DatetimeClose'].values,
                entry_price=trades['EntryPrice'].values,
                exit_price=trades['ExitPrice'].values,
                size=trades['Size'].values,
                pnl=trades['PnL'].values,
                pnl_comm=trades['PnLComm'].values,
                returns_pct=returns_pct.values,
                returns_positive=returns_positive.values,
            ))

            # Líneas de posición (desde entrada a salida)
            trade_source.add([[eb, xb] for eb, xb in zip(entry_bars, exit_bars)], 'position_lines_xs')
            trade_source.add([[ep, xp] for ep, xp in zip(trades['EntryPrice'], trades['ExitPrice'])], 'position_lines_ys')

            colors_darker = [_lightness(BEAR_COLOR, .35), _lightness(BULL_COLOR, .35)]
            trades_cmap = factor_cmap('returns_positive', colors_darker, ['0', '1'])

            fig_ohlc.multi_line(xs='position_lines_xs', ys='position_lines_ys',
                                source=trade_source, color=trades_cmap,
                                legend_label=f'Trades ({len(trades)})',
                                line_width=8, line_alpha=1, line_dash='dotted')

            # Panel P/L si procede
            if plot_pl:
                _log("[PLOT] Renderizando panel de P/L...")
                fig_pl = new_indicator_figure(y_axis_label="Profit / Loss", height=80)
                fig_pl.add_layout(Span(location=0, dimension='width', line_color='#666666',
                                       line_dash='dashed', level='underlay', line_width=1))

                size_abs = np.abs(trades['Size'])
                if size_abs.max() > size_abs.min():
                    marker_sizes = np.interp(size_abs, (size_abs.min(), size_abs.max()), (8, 20))
                else:
                    marker_sizes = np.full_like(size_abs, 10, dtype=float)
                trade_source.add(marker_sizes, 'marker_size')

                # Líneas verticales
                trade_source.add([[eb, xb] for eb, xb in zip(entry_bars, exit_bars)], 'lines')
                fig_pl.multi_line(xs='lines',
                                  ys=transform('returns_pct',
                                               CustomJSTransform(v_func='return [...xs].map(i => [0, i]);')),
                                  source=trade_source, color='#999', line_width=1)

                # Marcadores: triángulo si long, invertido si short
                triangles = np.where(trades['Size'] > 0, 'triangle', 'inverted_triangle')
                trade_source.add(triangles, 'triangles')

                cmap = factor_cmap('returns_positive', [BEAR_COLOR, BULL_COLOR], ['0', '1'])
                r_pl = fig_pl.scatter('exit_bar', 'returns_pct', source=trade_source,
                                      fill_color=cmap, marker='triangles',
                                      line_color='black', size='marker_size')

                set_tooltips(fig_pl, [("Size", "@size{0,0}"), ("P/L", "@returns_pct{+0.[000]%}")],
                             vline=False, renderers=[r_pl])
                fig_pl.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
                figs_above_ohlc.append(fig_pl)

        # ================== PANEL DE VOLUMEN ==================
        if plot_volume:
            _log("[PLOT] Renderizando panel de Volumen...")
            fig_volume = new_indicator_figure(height=70, y_axis_label="Volume")
            fig_volume.yaxis.ticker.desired_num_ticks = 3
            fig_volume.xaxis.formatter = fig_ohlc.xaxis[0].formatter
            fig_volume.xaxis.visible = True
            fig_ohlc.xaxis.visible = False  # enseñamos solo el eje x abajo

            r_vol = fig_volume.vbar('index', BAR_WIDTH, 'Volume', source=source, color=inc_cmap)
            set_tooltips(fig_volume, [('Volume', '@Volume{0.00 a}')], renderers=[r_vol])
            fig_volume.yaxis.formatter = NumeralTickFormatter(format="0 a")
            figs_below_ohlc.append(fig_volume)

        # Tooltips definitivos del OHLC (con datetime)
        set_tooltips(fig_ohlc, ohlc_tooltips, vline=True, renderers=[r_ohlc])

        # ================== INDICADORES ==================
        if indicators and len(indicators) > 0:
            _log(f"[PLOT] Renderizando {len(indicators)} indicadores...")
            colors = _colorgen()

            for ind_info in indicators:
                ind_name = ind_info.get('name', 'indicator')
                ind_values = ind_info.get('values', pd.Series(index=ohlcv.index, data=np.nan))
                is_overlay = bool(ind_info.get('overlay', True))

                # Opcionales
                has_marker = 'marker' in ind_info
                marker_type = ind_info.get('marker', 'circle')
                marker_color = ind_info.get('color', None)
                line_style = ind_info.get('line_style', 'solid')

                # Los indicadores ya están sincronizados con df.index (índice numérico)
                aligned = ind_values  # Ya está sincronizado, no necesita reindex
                key = f'ind_{ind_name}'
                source.add(aligned, key)

                if is_overlay:
                    _log(f"[PLOT]   - {ind_name} (overlay)")
                    color = marker_color if marker_color else next(colors)

                    if line_style == 'none' and has_marker:
                        r_ind = fig_ohlc.scatter('index', key, source=source,
                                                 marker=marker_type, size=10,
                                                 color=color, alpha=0.9,
                                                 legend_label=ind_name)
                    else:
                        r_ind = fig_ohlc.line('index', key, source=source,
                                              line_color=color, line_width=1.5,
                                              legend_label=ind_name, alpha=0.8)

                    # Incluir en extremos para autoscale dinámico
                    # y añadir tooltip por indicador
                    # Mostrar ms decimales en overlays (p.ej., zonas pivote) para evitar redondeos en tooltip
                    ohlc_tooltips.append((ind_name, f'@{key}{{0,0.0[000000]}}'))
                    # Acumulamos min/max por fila:
                    # (lo actualizamos al final para que entren todos los overlays)

                else:
                    _log(f"[PLOT]   - {ind_name} (oscillator)")
                    fig_osc = new_indicator_figure(y_axis_label=ind_name, height=70)
                    color = next(colors)
                    r_ind = fig_osc.line('index', key, source=source,
                                         line_color=color, line_width=1.3,
                                         legend_label=ind_name)
                    # Igual que overlays: mantener todos los decimales relevantes en los tooltips
                    set_tooltips(fig_osc, [(ind_name, f'@{key}{{0,0.0[000000]}}')], renderers=[r_ind])

                    # Líneas de referencia para RSI/ADX
                    if 'RSI' in ind_name.upper():
                        fig_osc.add_layout(Span(location=70, dimension='width',
                                                line_color='red', line_dash='dashed',
                                                line_width=1, line_alpha=0.5))
                        fig_osc.add_layout(Span(location=30, dimension='width',
                                                line_color='green', line_dash='dashed',
                                                line_width=1, line_alpha=0.5))
                        fig_osc.y_range = Range1d(0, 100)
                    elif 'ADX' in ind_name.upper():
                        fig_osc.add_layout(Span(location=20, dimension='width',
                                                line_color='gray', line_dash='dashed',
                                                line_width=1, line_alpha=0.5))
                        fig_osc.y_range = Range1d(0, 100)

                    figs_below_ohlc.append(fig_osc)

        # Actualizamos extremos para autoscale si hay overlays
        # (tomamos min y max por fila de todas las series registradas)
        extreme_min = df[['Low']].min(axis=1)
        extreme_max = df[['High']].max(axis=1)
        for k in list(source.data.keys()):
            if k.startswith('ind_'):
                s = pd.Series(source.data[k])
                # Asegurar que extreme_min y extreme_max sean arrays de numpy
                extreme_min_array = extreme_min.values if hasattr(extreme_min, 'values') else extreme_min
                extreme_max_array = extreme_max.values if hasattr(extreme_max, 'values') else extreme_max
                extreme_min = np.minimum(extreme_min_array, np.nan_to_num(s.values, nan=+np.inf))
                extreme_max = np.maximum(extreme_max_array, np.nan_to_num(s.values, nan=-np.inf))
        source.add(np.asarray(extreme_min), 'ohlc_low')
        source.add(np.asarray(extreme_max), 'ohlc_high')

        # Reaplicar tooltips OHLC (ya con indicadores añadidos)
        set_tooltips(fig_ohlc, ohlc_tooltips, vline=True, renderers=[r_ohlc])

        # Autoscale JS opcional
        if _AUTOSCALE_JS_CALLBACK:
            custom_js_args = dict(ohlc_range=fig_ohlc.y_range, source=source)
            if plot_volume and len(figs_below_ohlc) > 0:
                # volumen suele ser el primer fig_below_ohlc
                custom_js_args.update(volume_range=figs_below_ohlc[0].y_range)
            fig_ohlc.x_range.js_on_change('end', CustomJS(args=custom_js_args, code=_AUTOSCALE_JS_CALLBACK))

        # ================== ENSAMBLAR Y GUARDAR ==================
        _log("[PLOT] Ensamblando layout final...")

        figs = figs_above_ohlc + [fig_ohlc] + figs_below_ohlc

        # Crosshair compartido + estilo general
        linked_crosshair = CrosshairTool(dimensions='both', line_color='lightgrey')
        for f in figs:
            if getattr(f, 'legend', None):
                f.legend.visible = True
                f.legend.location = 'top_left'
                f.legend.border_line_width = 1
                f.legend.border_line_color = '#333333'
                f.legend.padding = 5
                f.legend.spacing = 0
                f.legend.margin = 0
                f.legend.label_text_font_size = '8pt'
                f.legend.click_policy = "hide"
                f.legend.background_fill_alpha = .9
            f.min_border_left = 0
            f.min_border_top = 3
            f.min_border_bottom = 6
            f.min_border_right = 10
            f.outline_line_color = '#666666'
            f.add_tools(linked_crosshair)
            # wheelzoom sin maintain_focus para UX suave
            wz = next((w for w in f.tools if isinstance(w, WheelZoomTool)), None)
            if wz:
                wz.maintain_focus = False

        fig = gridplot(figs, ncols=1, toolbar_location='right', toolbar_options=dict(logo=None), merge_tools=True)
        save(fig)
        _log(f"[PLOT] OK! Guardado en: {filename}")
        _log(f"[PLOT] Total paneles: {len(figs)}")

        return fig
    # Explicación extensa:
    # 1) Normalizamos columnas OHLCV y trades para aceptar los distintos formatos de tu pipeline (nuevo y antiguo).
    # 2) Creamos subpaneles (equity/drawdown/volumen/PL) solo si los datos existen.
    # 3) Para el eje X usamos índice lineal (posiciones) pero renderizamos etiquetas de tiempo con CustomJSTickFormatter.
    # 4) Los trades se dibujan como líneas punteadas entre (entrada->salida) en OHLC y un panel de P/L con triángulos.
    # 5) Indicadores: overlay (línea o marcadores) u oscillators (panel aparte). Añadimos tooltips por indicador.
    # 6) Autoscale por JS (si autoscale_cb.js está presente). Guardamos a HTML y devolvemos el layout para tests.


# =========================================================
# ==================== CLASE HEATMAPS =====================
# =========================================================
class Heatmaps:
    def generate(self, grid_df, param_cols: List[str], metric: str, filename: str, ncols: int = 2):
        """
        Crea heatmaps interactivos de TODAS las parejas de parámetros de 'param_cols'
        usando 'metric' como color. Guarda un único HTML con un grid de figuras.

        Parámetros:
          grid_df: DataFrame con columnas de parámetros + métrica.
          param_cols: lista de parámetros (>= 2).
          metric: nombre de la métrica a colorear (p.ej., 'Sharpe Ratio').
          filename: ruta de salida .html.
          ncols: nº de columnas en el grid (layout).

        Retorna:
          fig (layout de Bokeh) o None si no hay suficientes parámetros.
        """
        _log("\n[HEATMAP] Iniciando generación de heatmaps...")
        if not isinstance(param_cols, (list, tuple)) or len(param_cols) < 2:
            _log("[HEATMAP][ERROR] Se necesitan al menos 2 parámetros.")
            return None

        if not filename:
            filename = 'backtest_heatmap.html'
        if not filename.endswith('.html'):
            filename += '.html'
        output_file(filename, title=filename)

        # Validar parámetros presentes en grid_df
        valid_params = [p for p in param_cols if p in grid_df.columns]
        if len(valid_params) < 2:
            _log(f"[HEATMAP][ERROR] Param cols no presentes en grid_df: {param_cols}")
            return None

        # Generar todas las combinaciones de 2 parámetros
        pairs = list(combinations(valid_params, 2))
        _log(f"[HEATMAP] Grid: {len(grid_df)} combinaciones | Parejas a plotear: {pairs} | Métrica: {metric}")

        figs = []
        plot_width = 1200  # fijo interno (antes se pasaba como arg). Ajusta si lo necesitas.

        # Rango global de colores (consistente entre figuras)
        if metric not in grid_df.columns:
            _log(f"[HEATMAP][ERROR] La métrica '{metric}' no existe en grid_df.")
            return None

        global_min = pd.to_numeric(grid_df[metric], errors='coerce').min()
        global_max = pd.to_numeric(grid_df[metric], errors='coerce').max()

        cmap = LinearColorMapper(
            palette=Viridis256,
            low=global_min,
            high=global_max,
            nan_color='white'
        )

        # Crear cada heatmap (pivot -> melt -> rect)
        for p1, p2 in pairs:
            _log(f"[HEATMAP] Procesando {p1} x {p2}...")

            pivot = grid_df.pivot_table(index=p2, columns=p1, values=metric, aggfunc='mean')
            pivot_reset = pivot.reset_index()
            pivot_long = pivot_reset.melt(id_vars=p2, var_name=p1, value_name='_Value')

            # Ejes categóricos ordenados por valor numérico
            pivot_long[p1] = pivot_long[p1].astype(str)
            pivot_long[p2] = pivot_long[p2].astype(str)
            try:
                level1 = sorted(pivot_long[p1].unique(), key=lambda x: float(x))
                level2 = sorted(pivot_long[p2].unique(), key=lambda x: float(x))
            except Exception:
                # Si no es convertible a float, orden alfabético
                level1 = sorted(pivot_long[p1].unique())
                level2 = sorted(pivot_long[p2].unique())

            fig = _figure(
                x_range=level1, y_range=level2,
                x_axis_label=p1, y_axis_label=p2,
                width=plot_width // ncols, height=plot_width // ncols,
                tools='box_zoom,reset,save,hover',
                tooltips=[(p1, f'@{p1}'), (p2, f'@{p2}'), (metric, '@_Value{0.[000]}')]
            )

            fig.grid.grid_line_color = None
            fig.axis.axis_line_color = None
            fig.axis.major_tick_line_color = None
            fig.axis.major_label_standoff = 0
            fig.xaxis.major_label_orientation = 45

            fig.rect(x=p1, y=p2, width=1, height=1, source=pivot_long, line_color=None,
                     fill_color=dict(field='_Value', transform=cmap))
            figs.append(fig)

        # Barra de color en la primera figura
        if len(figs) > 0:
            color_bar = ColorBar(color_mapper=cmap, ticker=BasicTicker(desired_num_ticks=10),
                                 label_standoff=12, border_line_color=None, location=(0, 0))
            figs[0].add_layout(color_bar, 'right')

        fig = gridplot(figs, ncols=ncols, toolbar_location='above', toolbar_options=dict(logo=None), merge_tools=True)
        save(fig)
        _log(f"[HEATMAP] OK! Guardado en: {filename}")
        _log(f"[HEATMAP] Total heatmaps: {len(figs)}")

        return fig
    # Explicación:
    # Este método genera un conjunto de heatmaps para todas las parejas de 'param_cols'
    # sobre la métrica elegida. El rango de color es global para facilitar comparación.
    # Se usa pivot_table + melt para dibujar con rectángulos categóricos.

