# ================================================================
# backtest.py — Núcleo de ejecución y optimización con Backtrader
# ================================================================
# Qué aporta este módulo (resumen):
#   1) Clase Backtest:
#        - run_backtrader_core(df_master, strategy_cls, params,
#                              return_trades=True, return_metrics=True, return_indicators=True)
#          -> (df_trades, df_metrics, df_indicators)  SIEMPRE DataFrames
#
#          df_trades:  open_datetime, close_datetime, size, price_open, price_close, pnl, pnl_comm, barlen
#          df_metrics: métricas unificadas (Equity Final, Peak, Buy&Hold, Return Ann, Vol Ann, Sharpe, Sortino,
#                     Drawdowns, PF, Expectancy, SQN, WinRate, etc.)
#          df_indicators: serie temporal con equity y drawdown (+cualquier indicador extra que exporte la estrategia)
#
#   2) Clase OptimizeCore:
#        - optimize(backtest, strategy_cls, df_master, param_cols=None, ranges=None)
#             -> grid_df con (parámetros + todas las métricas recolectadas por run_backtrader_core)
#             -> respeta restricciones de config.OPTIMIZE_CONSTRAINT
#        - select_best_params(grid_df, backtest, strategy_cls, df_master, param_cols=None)
#             -> usa config.OPTIMIZE_MAXIMIZE y sentido de ranking de metrics.minimize_metrics()
#             -> recalcula la mejor combinación y devuelve (df_trades, df_metrics, df_indicators)
#
#   3) Clases auxiliares de salida:
#        - Grid.save(grid_df, strategy_name, suffix="")
#        - Heatmaps.make(grid_df, param_cols, metric, strategy_name)
#        - Stats.save(metrics_df, strategy_name, params=None)
#        - Trades.save(trades_df, strategy_name, params=None)
#        - Plots.make(ohlcv, trades_df, indicators_df, strategy_name, filename="")
#        - Results.build_summary(order_by=config.SUMMARY_FIRST_COL)
#
# Notas:
#   - Este módulo no lee CSV: DataManager (data.py) se encarga de cargar/sincronizar y tú pasas df_master = [df01, df02, ...]
#   - Respeta tus toggles en config.py (MARGIN, COMMISSION, TRADE_ON_CLOSE, OPTIMIZE_MAXIMIZE, etc.)
#   - Logs MUY verbosos (comenta los print si molestan).
# ================================================================

from __future__ import annotations

import os
import itertools
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import backtrader as bt

import config
from metrics import MetricsCalculator, minimize_metrics           # :contentReference[oaicite:0]{index=0}
from plotting import Plot, Heatmaps as PlotHeatmaps              # :contentReference[oaicite:1]{index=1}
from data import safe_name, ensure_dirs, output_path         # :contentReference[oaicite:2]{index=2}


# -------------------------
# LOG global muy verboso
# -------------------------
def _log(msg: str) -> None:
    print(msg)


# ================================================================
# =========================  BACKTEST  ===========================
# ================================================================
class Backtest:
    """
    Encapsula la ejecución de una estrategia Backtrader sobre 1..N DataFrames ya cargados/sincronizados.
    Devuelve SIEMPRE tres DataFrames: trades, metrics y indicators (equity/drawdown + opcionales).
    """

    # --------------------- CommInfo por % + margen ---------------------
    @staticmethod
    def _make_commission_info(commission_pct: float, margin_factor: float):
        """
        Crea una CommInfoBase con comisión en porcentaje y "margin" interpretado como
        FRACCIÓN de margen requerido (es decir, leverage = 1/margin).
        Con tu toggle: MARGIN = 5  -> leverage 1:5 -> margin = 1/5 = 0.2

        Devuelve:
            bt.CommInfoBase subclass instance
        """
        class _PctCommInfo(bt.CommInfoBase):
            params = dict(
                commission=commission_pct,
                commtype=bt.CommInfoBase.COMM_PERC,
                margin=margin_factor if margin_factor > 0 else 1.0,
                mult=1.0
            )
        return _PctCommInfo()

    def __init__(self, broker_cfg: Optional[Dict[str, Any]] = None):
        """
        broker_cfg opcional: {"cash": float, "commission": float, "margin": float, "trade_on_close": bool}
        Si no pasas nada, toma de config los valores clave (commission, margin, trade_on_close)
        y un cash por defecto de 100000.0.
        """
        broker_cfg = broker_cfg or {}
        # margin en Backtrader será 1 / MARGIN (porque MARGIN es el apalancamiento 1:X)
        margin_from_config = 1.0 / max(1.0, float(getattr(config, "MARGIN", 1)))
        commission_from_config = float(getattr(config, "COMMISSION", 0.0)) / 100.0 # 0.0025% -> 0.0025
        coc_from_config = bool(getattr(config, "TRADE_ON_CLOSE", True))

        self.broker_cfg = {
            "cash": float(broker_cfg.get("cash", 100000.0)),
            "commission": float(broker_cfg.get("commission", commission_from_config)),
            "margin": float(broker_cfg.get("margin", margin_from_config)),
            "trade_on_close": bool(broker_cfg.get("trade_on_close", coc_from_config))
        }
        _log(f"[Backtest] Broker config: {self.broker_cfg}")

    # ----------------------- helper: inferir PPA -----------------------
    def _infer_periods_per_year(self, df: pd.DataFrame) -> int:
        """
        Estima 'periods_per_year' en función del spacing temporal del primer DataFrame.
        - Diario o superior: 252
        - Horario ~ 24*252
        - Intradía: aproximación por minutos -> 60*24*252/mins
        """
        try:
            if not isinstance(df.index, pd.DatetimeIndex) or len(df.index) < 2:
                return 252
            deltas = np.diff(df.index.view('i8'))  # ns entre barras
            if len(deltas) == 0:
                return 252
            median_ns = np.median(deltas)
            minutes = median_ns / (1e9 * 60.0)
            if minutes >= 60:
                return 24 * 252
            if minutes >= 1:
                return int((60 * 24 * 252) / max(1, int(round(minutes))))
            return 24 * 252
        except Exception:
            return 252

    # --------------------- núcleo de ejecución -------------------------
    def run_backtrader_core(
        self,
        df_master: List[pd.DataFrame],         # lista plana: [df01, df02, df03, ...]
        strategy_cls,                          # clase de estrategia (NO instancia)
        params: Optional[Dict[str, Any]] = None,
        return_trades: bool = True,
        return_metrics: bool = True,
        return_indicators: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Ejecuta Backtrader con los DataFrames pasados en df_master y devuelve SIEMPRE:
            (df_trades, df_metrics, df_indicators) como DataFrames.
        Maneja multi-activo: añade todos los DataFrames que no estén vacíos.
        """
        _log("\n[Backtest] ====== INICIO run_backtrader_core ======")

        # 1) Cerebro + broker
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.broker.setcash(self.broker_cfg["cash"])
        cerebro.broker.set_coc(self.broker_cfg["trade_on_close"])
        comminfo = self._make_commission_info(
            commission_pct=self.broker_cfg["commission"],
            margin_factor=self.broker_cfg["margin"]
        )
        cerebro.broker.addcommissioninfo(comminfo)
        _log("[Backtest] Cerebro y broker configurados")

        # 2) Añadir DataFeeds de Pandas (acepta 1..N)
        first_df = None
        datas_added = 0
        for i, df in enumerate(df_master):
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                _log(f"[Backtest][INFO] df_master[{i}] vacío o inválido -> ignorado")
                continue
            dname = f"data{i}"
            datafeed = bt.feeds.PandasData(
                dataname=df,
                open='open', high='high', low='low', close='close', volume='volume',
                datetime=None, timeframe=bt.TimeFrame.Minutes
            )
            datafeed._name = dname
            # Etiquetas auxiliares para trazas de eventos (paridad)
            tf_label = "unknown"
            try:
                if len(df.index) >= 2:
                    delta_min = (df.index[1] - df.index[0]).total_seconds() / 60.0
                    tf_label = f"M{int(round(delta_min))}" if delta_min >= 1 else "S"
            except Exception:
                pass
            datafeed._symbol = getattr(df, "_asset_name", dname)
            datafeed._timeframe_label = tf_label
            cerebro.adddata(datafeed)
            if first_df is None:
                first_df = df
            datas_added += 1
            _log(f"[Backtest] Añadido {dname} con {len(df)} velas")

        if datas_added == 0:
            _log("[Backtest][ERROR] df_master no contiene ningún DataFrame válido")
            empty_trades = pd.DataFrame(columns=["open_datetime", "close_datetime", "size",
                                                 "price_open", "price_close", "pnl", "pnl_comm", "barlen"])
            return empty_trades, pd.DataFrame(), pd.DataFrame()

        # 2.1) Auto-resample para PivotZoneTest cuando solo llega data M1 en DATA_FOLDER_01.
        #      Objetivo: permitir usar un único CSV M1 y construir TF_zone=M3 internamente.
        strategy_name = getattr(strategy_cls, "__name__", str(strategy_cls))
        if datas_added == 1 and strategy_name == "PivotZoneTest":
            base_feed = cerebro.datas[0]
            base_minutes = 1.0
            try:
                if first_df is not None and len(first_df.index) >= 2:
                    base_minutes = (first_df.index[1] - first_df.index[0]).total_seconds() / 60.0
            except Exception:
                base_minutes = 1.0

            if base_minutes <= 0:
                base_minutes = 1.0

            target_zone_minutes = 3.0
            compression = int(round(target_zone_minutes / base_minutes))
            if compression > 1:
                zone_feed = cerebro.resampledata(
                    base_feed,
                    timeframe=bt.TimeFrame.Minutes,
                    compression=compression,
                    name="data1",
                )
                zone_feed._name = "data1"
                zone_feed._symbol = getattr(base_feed, "_symbol", "data0")
                zone_feed._timeframe_label = f"M{int(round(target_zone_minutes))}"
                _log(
                    "[Backtest] Auto-resample PivotZoneTest activado: "
                    f"base=M{base_minutes:.0f} -> zone=M{int(round(target_zone_minutes))} (compression={compression})"
                )
            else:
                _log(
                    "[Backtest][WARN] Auto-resample PivotZoneTest omitido: "
                    f"base timeframe ~M{base_minutes:.0f} no permite generar M3 con compression>1"
                )

        # 3) Estrategia y parámetros
        params = params or {}
        cerebro.addstrategy(strategy_cls, **params)
        _log(f"[Backtest] Estrategia: {getattr(strategy_cls, '__name__', str(strategy_cls))} | params={params}")

        # 4) Analyzers mínimos (retornos por barra, DD)
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='tr')  # Sin timeframe específico.
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')

        # 5) Ejecutar
        results = cerebro.run(runonce=False, preload=True)
        strat = results[0]
        cash0 = float(self.broker_cfg["cash"])
        _log("[Backtest] Ejecución completada")

        # 6) Equity curve y drawdown de la cuenta (sanitizando fechas del analizador)
        tr_dict = strat.analyzers.tr.get_analysis()  # {datetime->ret}
        returns = pd.Series(tr_dict).sort_index()

        # Limitar fechas al rango real de datos para evitar centinelas (p.ej., año 9999)
        lo = first_df.index.min() if first_df is not None and len(first_df.index) else None
        hi = first_df.index.max() if first_df is not None and len(first_df.index) else None
        dt = pd.to_datetime(returns.index, errors="coerce")
        if lo is not None and hi is not None:
            mask = dt.notna() & (dt >= lo) & (dt <= hi)
        else:
            mask = dt.notna()
        returns = returns[mask]
        dt = dt[mask]

        if returns.empty:
            equity = pd.Series(dtype=float, name="equity")
            drawdown = pd.Series(dtype=float, name="drawdown")
        else:
            returns.index = dt
            equity = (1.0 + returns.fillna(0.0)).cumprod() * cash0
            equity.name = "equity"

            run_max = equity.cummax()
            drawdown = (equity - run_max) / run_max
            drawdown.name = "drawdown"

        # 7) Reconstruir trades (estrategia base guarda self._closed_trades)
        if hasattr(strat, "_closed_trades") and isinstance(strat._closed_trades, list):
            df_trades = pd.DataFrame(strat._closed_trades)
        else:
            df_trades = pd.DataFrame(columns=["open_datetime", "close_datetime", "size",
                                              "price_open", "price_close", "pnl", "pnl_comm", "barlen"])
        _log(f"[Backtest] Trades reconstruidos: {len(df_trades)}")

        # 8) Métricas (fusionando analizador + calculadora propia)
        metrics = {}
        # Dejamos que MetricsCalculator compute Sharpe/Sortino/AnnReturn/AnnVol con geometría estilo backtesting.py
        metrics["Equity Final [$]"] = float(equity.iloc[-1]) if len(equity) else np.nan

        # Completar con MetricsCalculator (drawdowns, sortino, retorno anual, etc.)
        calc = MetricsCalculator()
        m_more = calc.calculate_all(
            equity_curve=equity,
            ohlcv=first_df,
            trades_df=df_trades if not df_trades.empty else None,
            initial_cash=cash0,
            returns_series=returns,
            periods_per_year=self._infer_periods_per_year(first_df)
        )
        metrics.update(m_more)
        df_metrics = pd.DataFrame([metrics])
        _log(f"[Backtest] Métricas calculadas -> columnas: {list(df_metrics.columns)}")

        # 9) Indicators (equity + drawdown + opcionales exportados por estrategia)
        # IMPORTANTE:
        # - En algunos entornos (versiones de Backtrader/Pandas), la serie de retornos del analyzer
        #   puede tener 1..N puntos menos que el DataFrame OHLCV base.
        # - Los indicadores exportados desde Backtrader (`line.array`) suelen tener longitud ~= len(data)
        #   (incluyendo warmup con NaN), y por tanto pueden venir MÁS largos que `equity`.
        # - En Windows esto puede “casar” por casualidad, pero en Linux/VPS es común ver desajustes
        #   y que falle el armado de `df_indicators`, dejando las líneas (p.ej. zonas pivote) fuera del plot.
        #
        # Decisión:
        # - Usamos el índice del primer OHLCV (`first_df.index`) como índice canónico para plotting/export.
        # - Alineamos equity/drawdown y cada indicador exportado a ese índice, recortando/paddeando si hace falta.
        base_index = first_df.index if (first_df is not None and hasattr(first_df, "index")) else equity.index

        def _align_values_to_index(values: Any, index: pd.Index, name: str) -> pd.Series:
            """
            Alinea una secuencia (list/np.array/Series) al `index` objetivo:
              - Si viene más larga -> recorta por la izquierda (conserva el tramo más reciente).
              - Si viene más corta -> rellena por la izquierda con NaN (para no desplazar el final).
            """
            if values is None:
                return pd.Series(index=index, dtype=float, name=name)
            try:
                seq = list(values)
            except Exception:
                _log(f"[Backtest][WARN] Indicador {name}: no se pudo convertir a lista -> se ignora")
                return pd.Series(index=index, dtype=float, name=name)

            n = len(seq)
            m = len(index)
            if n == 0 or m == 0:
                return pd.Series(index=index, dtype=float, name=name)

            if n > m:
                _log(f"[Backtest][WARN] Indicador {name}: len={n} > base_index={m} -> recorto izquierda")
                seq = seq[-m:]
            elif n < m:
                _log(f"[Backtest][WARN] Indicador {name}: len={n} < base_index={m} -> pad NaN izquierda")
                seq = ([np.nan] * (m - n)) + seq

            return pd.Series(seq, index=index, name=name)

        equity_plot = pd.Series(equity, copy=False)
        drawdown_plot = pd.Series(drawdown, copy=False)
        try:
            equity_plot = equity_plot.reindex(base_index).ffill()
            drawdown_plot = drawdown_plot.reindex(base_index).ffill()
            # Si al principio quedan NaN (por warmup o retornos incompletos), rellenamos hacia atrás para plot.
            if equity_plot.isna().any():
                equity_plot = equity_plot.bfill()
            if drawdown_plot.isna().any():
                drawdown_plot = drawdown_plot.bfill()
        except Exception as e:
            _log(f"[Backtest][WARN] No se pudo reindexar equity/drawdown a base_index: {e}")
            equity_plot = equity
            drawdown_plot = drawdown

        ind = pd.DataFrame({"equity": equity_plot, "drawdown": drawdown_plot}, index=base_index)
        indicator_colors = {}  # Metadata de colores para plotting
        if hasattr(strat, "export_indicators") and callable(strat.export_indicators):
            try:
                extra = strat.export_indicators() or {}
                _log(f"[Backtest] Indicadores exportados: {list(extra.keys())}")
                
                # Extraer metadata de colores si existe
                if '_colors' in extra:
                    indicator_colors = extra.pop('_colors')
                    _log(f"[Backtest] Colores personalizados: {list(indicator_colors.keys())}")
                
                for k, v in extra.items():
                    # Saltar keys de metadata (empiezan con _)
                    if k.startswith('_'):
                        continue
                    if v is None:
                        _log(f"[Backtest][WARN] Indicador {k} es None -> ignorado")
                        continue

                    # Evitar `if v` porque con numpy arrays puede lanzar ValueError (truth value ambiguous)
                    try:
                        v_len = len(v)
                    except Exception:
                        _log(f"[Backtest][WARN] Indicador {k} sin len() -> ignorado")
                        continue

                    if v_len <= 0:
                        _log(f"[Backtest][WARN] Indicador {k} vacío (len=0) -> ignorado")
                        continue

                    ser = _align_values_to_index(v, base_index, name=str(k))
                    ind[str(k)] = ser.values
                    _log(f"[Backtest] Indicador {k} agregado: len_in={v_len} len_out={len(ser)}")
            except Exception as e:
                _log(f"[Backtest][WARN] export_indicators falló: {e}")
                import traceback
                _log(f"[Backtest][WARN] Traceback: {traceback.format_exc()}")
        df_indicators = ind.copy()
        df_indicators.attrs['_colors'] = indicator_colors  # Guardar colores como metadata
        _log(f"[Backtest] Indicators listos: {list(df_indicators.columns)}")

        _log("[Backtest] ====== FIN run_backtrader_core ======\n")
        return df_trades, df_metrics, df_indicators

    # Comentarios (lectura para novato):
    # - _make_commission_info: Backtrader usa 'margin' como fracción requerida. Si pones margin=0.2, simulas 1:5.
    # - __init__: recoge los toggles de tu config (COMMISSION, MARGIN, TRADE_ON_CLOSE) y los aplica al broker.
    # - run_backtrader_core:
    #     1) Monta Cerebro y broker (cash, comisión %, apalancamiento vía margin, trade_on_close).
    #     2) Añade TODOS los DataFrames no vacíos como feeds (sirve para multi-activo).
    #     3) Añade la estrategia con los parámetros recibidos.
    #     4) Inserta analyzers básicos: TimeReturn (serie de retornos) y DrawDown.
    #     5) Ejecuta y recupera:
    #           - equity = capital acumulado (usando retornos -> cumprod).
    #           - drawdown = (equity - max_acum) / max_acum.
    #           - trades = lista reconstruida por la base de tus estrategias.
    #     6) Calcula Sharpe y llama a MetricsCalculator para el resto de métricas y trade-stats.
    #     7) indicators_df = equity + drawdown (+lo que exporte la estrategia opcionalmente).
    #   Devuelve SIEMPRE tres DataFrames listos para guardar/plotear/evaluar.


# ================================================================
# =======================  OPTIMIZE CORE  ========================
# ================================================================
class OptimizeCore:
    """
    Recorre un grid de parámetros y construye un DataFrame con:
      - columnas de parámetros (param_cols)
      - columnas de métricas devueltas por Backtest.run_backtrader_core (df_metrics)
    Usa config.OPTIMIZE_CONSTRAINT (expresión tipo 'n1 < n2 and n2 < n3') para filtrar combinaciones.
    """

    def __init__(self, maximize_metric: Optional[str] = None):
        self.maximize_metric = maximize_metric or getattr(config, "OPTIMIZE_MAXIMIZE", None)
        _log(f"[OptimizeCore] Métrica objetivo: {self.maximize_metric}")

    # ------------------- helpers de ranking/constraint -------------------
    def _rank_value(self, metric_name: str, value: Any) -> float:
        """
        Convierte cualquier métrica a un valor donde "MÁS ALTO = MEJOR".
        Si la métrica pertenece a minimize_metrics() (drawdowns, volatilidad...), invierte el signo.
        """
        try:
            x = float(value)
        except Exception:
            return -np.inf  # NaN o no numérico -> pésimo

        if metric_name in minimize_metrics():  # :contentReference[oaicite:3]{index=3}
            if np.isnan(x):
                return -np.inf
            return -x  # menor es mejor -> invertimos
        else:
            if np.isnan(x):
                return -np.inf
            return x

    def _satisfies_constraint(self, combo: Dict[str, Any], expr: Optional[str]) -> bool:
        """
        Evalúa de forma acotada una expresión de restricción sobre 'combo'.
        Ej: expr='n1 < n2 and n2 < n3'
        """
        if not expr:
            return True
        try:
            safe_locals = combo
            return bool(eval(expr, {"__builtins__": {}}, safe_locals))
        except Exception as e:
            _log(f"[OptimizeCore][WARN] Constraint '{expr}' falló con {combo}: {e}")
            return True  # si falla, no bloqueamos

    # ------------------------- optimize() -------------------------------
    def optimize(
        self,
        backtest: Backtest,
        strategy_cls,
        df_master: List[pd.DataFrame],
        param_cols: Optional[List[str]] = None,
        ranges: Optional[Dict[str, List[Any]]] = None
    ) -> pd.DataFrame:
        """
        Genera grid de parámetros y ejecuta la estrategia por combinación válida.
        Devuelve grid_df con una fila por combinación y columnas:
          param_cols + TODAS las métricas agregadas por run_backtrader_core().
        """
        _log("\n[OptimizeCore] ====== INICIO optimize ======")
        param_cols = param_cols or list(getattr(config, "param_cols", ['n1', 'n2', 'n3']))
        # Construcción de rangos por defecto desde config
        if ranges is None:
            ranges = {}
            if 'n1' in param_cols: ranges['n1'] = list(getattr(config, "N1_RANGE", []))
            if 'n2' in param_cols: ranges['n2'] = list(getattr(config, "N2_RANGE", []))
            if 'n3' in param_cols: ranges['n3'] = list(getattr(config, "N3_RANGE", []))

        # Producto cartesiano en el orden de param_cols
        grid_values = [ranges.get(p, [None]) for p in param_cols]
        combos = list(itertools.product(*grid_values))
        _log(f"[OptimizeCore] Combinaciones generadas: {len(combos)}")

        rows: List[Dict[str, Any]] = []
        constraint = getattr(config, "OPTIMIZE_CONSTRAINT", None)

        for idx, values in enumerate(combos):
            combo = {p: int(v) if p in ['n1', 'n2', 'n3', 'adx_period', 'rsi_period'] else v for p, v in zip(param_cols, values)}
            if not self._satisfies_constraint(combo, constraint):
                _log(f"[OptimizeCore] ({idx+1}/{len(combos)}) {combo} -> descartado por constraint")
                continue

            _log(f"[OptimizeCore] ({idx+1}/{len(combos)}) Ejecutando con {combo}")
            df_tr, df_me, df_ind = backtest.run_backtrader_core(
                df_master=df_master,
                strategy_cls=strategy_cls,
                params=combo,
                return_trades=True,
                return_metrics=True,
                return_indicators=True
            )

            # compactar métricas (una fila) + parámetros
            row = {**combo}
            if not df_me.empty:
                for k, v in df_me.iloc[0].to_dict().items():
                    row[k] = v
            else:
                row["Equity Final [$]"] = np.nan
                row["Sharpe Ratio"] = np.nan
            rows.append(row)

        if not rows:
            _log("[OptimizeCore][WARN] No se generaron filas (grid vacío tras constraint?)")
            return pd.DataFrame(columns=param_cols + ["Sharpe Ratio"])

        grid_df = pd.DataFrame(rows)
        # Columna de ranking opcional para inspección rápida
        if self.maximize_metric and self.maximize_metric in grid_df.columns:
            grid_df["__rank__"] = grid_df[self.maximize_metric].apply(
                lambda x: self._rank_value(self.maximize_metric, x)
            )
        _log(f"[OptimizeCore] Grid listo con {len(grid_df)} filas y {len(grid_df.columns)} columnas")
        _log("[OptimizeCore] ====== FIN optimize ======\n")
        return grid_df

    # --------------------- select_best_params() -------------------------
    def select_best_params(
        self,
        grid_df: pd.DataFrame,
        backtest: Backtest,
        strategy_cls,
        df_master: List[pd.DataFrame],
        param_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Selecciona la mejor combinación según config.OPTIMIZE_MAXIMIZE y su sentido (min/max).
        Recalcula la estrategia con esos parámetros y devuelve:
            (df_trades, df_metrics, df_indicators)
        """
        _log("\n[OptimizeCore] ====== INICIO select_best_params ======")
        metric = self.maximize_metric or getattr(config, "OPTIMIZE_MAXIMIZE", None)
        if not metric or metric not in grid_df.columns:
            raise ValueError(f"[OptimizeCore] Métrica objetivo inválida o ausente en grid_df: {metric}")

        # Ranking: mayor _rank_value = mejor (invierte signo si 'menor es mejor')
        scores = grid_df[metric].apply(lambda x: self._rank_value(metric, x))
        best_idx = int(scores.idxmax())
        best_row = grid_df.loc[best_idx]
        _log(f"[OptimizeCore] Mejor combinación (fila {best_idx}): {best_row.to_dict()}")

        # Extraer parámetros desde param_cols
        param_cols = param_cols or list(getattr(config, "param_cols", ['n1', 'n2', 'n3']))
        params = {p: int(best_row[p]) for p in param_cols if p in best_row.index}

        # Recalcular
        df_trades, df_metrics, df_indicators = backtest.run_backtrader_core(
            df_master=df_master,
            strategy_cls=strategy_cls,
            params=params,
            return_trades=True,
            return_metrics=True,
            return_indicators=True
        )
        _log("[OptimizeCore] Recalculo de la mejor combinación completado")
        _log("[OptimizeCore] ====== FIN select_best_params ======\n")
        return df_trades, df_metrics, df_indicators

    # Comentarios (lectura para novato):
    # - optimize():
    #     * Construye producto cartesiano con los rangos de config (N1_RANGE, N2_RANGE, N3_RANGE).
    #     * Filtra por OPTIMIZE_CONSTRAINT (si hay) usando eval protegido con locals=combo.
    #     * Ejecuta la estrategia para cada combinación y guarda métricas en filas.
    # - select_best_params():
    #     * Usa OPTIMIZE_MAXIMIZE. Si la métrica es de tipo "menor es mejor", _rank_value invierte el signo.
    #     * Elige la mejor, RE-EJECUTA run_backtrader_core y te devuelve los tres DataFrames para guardar/plotear.


# ================================================================
# ===============  CONTENEDORES / SERVICIOS DE SALIDA  ===========
# ================================================================
class Grid:
    """Guarda el grid de optimización en /grid en .xlsx"""
    @staticmethod
    def save(grid_df: pd.DataFrame, strategy_name: str, suffix: str = "") -> str:
        ensure_dirs()
        fname = output_path(f"grid/{safe_name(strategy_name)}{('_' + suffix) if suffix else ''}.xlsx")
        grid_df.to_excel(fname, index=False)
        _log(f"[Grid] Guardado: {fname}")
        return fname


class Heatmaps:
    """Genera Heatmaps interactivos usando plotting.Heatmaps.generate(...)"""
    @staticmethod
    def make(grid_df: pd.DataFrame, param_cols: List[str], metric: str, strategy_name: str) -> str:
        ensure_dirs()
        filename = output_path(f"heatmaps/{safe_name(strategy_name)}_{safe_name(metric)}.html")
        PlotHeatmaps().generate(grid_df, param_cols=param_cols, metric=metric, filename=filename)  # :contentReference[oaicite:4]{index=4}
        _log(f"[Heatmaps] Guardado: {filename}")
        return filename


class Stats:
    """Guarda el DataFrame de métricas (una fila) en /stats con los parámetros como columnas extra si se pasan."""
    @staticmethod
    def save(metrics_df: pd.DataFrame, strategy_name: str, params: Optional[Dict[str, Any]] = None) -> str:
        ensure_dirs()
        df = metrics_df.copy()
        if params:
            for k, v in params.items():
                df[k] = v
        fname = output_path(f"stats/{safe_name(strategy_name)}.xlsx")
        df.to_excel(fname, index=False)
        _log(f"[Stats] Guardado: {fname}")
        return fname


class Trades:
    """Guarda los trades en /trades"""
    @staticmethod
    def save(trades_df: pd.DataFrame, strategy_name: str, params: Optional[Dict[str, Any]] = None) -> str:
        ensure_dirs()
        df = trades_df.copy()

        # Calcular el tamaño monetario invertido por trade (size × precio de apertura).
        # Se detectan las columnas de forma insensible a mayúsculas para mantener compatibilidad.
        size_col = next((c for c in df.columns if c.lower() == "size"), None)
        price_col = next((c for c in df.columns if c.lower() in {"price_open", "open_price"}), None)

        if size_col and price_col:
            df["size_equity"] = (
                pd.to_numeric(df[size_col], errors="coerce")
                * pd.to_numeric(df[price_col], errors="coerce")
            )

        if params:
            for k, v in params.items():
                df[k] = v
        fname = output_path(f"trades/{safe_name(strategy_name)}.xlsx")
        df.to_excel(fname, index=False)
        _log(f"[Trades] Guardado: {fname}")
        return fname

class CSCV:
    """Guarda los resultados del análisis CSCV en /cscv"""
    @staticmethod
    def save(cscv_df: pd.DataFrame, strategy_name: str, suffix: str = "") -> str:
        ensure_dirs()
        df = cscv_df.copy()
        fname = output_path(f"cscv/{safe_name(strategy_name)}{('_' + suffix) if suffix else ''}.xlsx")
        df.to_excel(fname, index=False)
        _log(f"[CSCV] Guardado: {fname}")
        return fname

class Stress:
    """Guarda los resultados del análisis Stress Test en /stress"""
    @staticmethod
    def save(stress_df: pd.DataFrame, strategy_name: str, suffix: str = "") -> str:
        ensure_dirs()
        df = stress_df.copy()
        fname = output_path(f"stress/{safe_name(strategy_name)}{('_' + suffix) if suffix else ''}.xlsx")
        df.to_excel(fname, index=False)
        _log(f"[Stress] Guardado: {fname}")
        return fname

class WF:
    """Guarda los resultados del análisis Walk-Forward en /wf"""
    @staticmethod
    def save(wf_df: pd.DataFrame, strategy_name: str, suffix: str = "") -> str:
        ensure_dirs()
        df = wf_df.copy()
        fname = output_path(f"wf/{safe_name(strategy_name)}{('_' + suffix) if suffix else ''}.xlsx")
        df.to_excel(fname, index=False)
        _log(f"[WF] Guardado: {fname}")
        return fname

class Plots:
    """Crea el HTML interactivo del backtest con velas, equity y drawdown."""
    @staticmethod
    def make(ohlcv: pd.DataFrame, trades_df: pd.DataFrame, indicators_df: pd.DataFrame,
             strategy_name: str, filename: str = "") -> str:
        ensure_dirs()
        if not filename:
            filename = output_path(f"plots/{safe_name(strategy_name)}.html")
        elif not os.path.isabs(filename):
            filename = output_path(filename)
        
        # Extraer equity
        equity = indicators_df["equity"] if ("equity" in indicators_df.columns) else None
        
        # Convertir indicadores a formato esperado por max_min_plot
        indicators_list = []
        # Leer metadata de colores si existe
        indicator_colors = indicators_df.attrs.get('_colors', {})
        if indicator_colors:
            _log(f"[Plots] Colores personalizados disponibles: {list(indicator_colors.keys())}")
        
        if not indicators_df.empty:
            _log(f"[Plots] Procesando indicators_df con columnas: {list(indicators_df.columns)}")
            for col in indicators_df.columns:
                if col not in ['equity', 'drawdown']:  # Excluir equity y drawdown
                    # Determinar si es overlay o panel separado
                    is_overlay = True  # Por defecto overlay
                    if col.upper() in ['ADX', 'RSI', 'MACD', 'STOCH', 'RSI_SMA']:  # Osciladores conocidos
                        is_overlay = False
                    
                    # Usar color personalizado si existe
                    custom_color = indicator_colors.get(col, None)
                    
                    indicators_list.append({
                        'name': col,
                        'values': indicators_df[col],
                        'overlay': is_overlay,
                        'color': custom_color,  # Usar color personalizado o None
                        'line_style': 'solid'
                    })
                    _log(f"[Plots] Indicador {col} agregado (overlay={is_overlay}, color={custom_color})")
        else:
            _log(f"[Plots][WARN] indicators_df está vacío")
        
        _log(f"[Plots] Total indicadores para plot: {len(indicators_list)}")
        
        Plot().max_min_plot(
            ohlcv=ohlcv, 
            equity_curve=equity, 
            trades_df=trades_df, 
            indicators=indicators_list,  # Pasar la lista de indicadores
            filename=filename
        )
        _log(f"[Plots] Guardado: {filename}")
        return filename


class Results:
    """
    Compila los .xlsx de /stats en un summary ordenado por config.SUMMARY_FIRST_COL
    y lo guarda en /results/summary_stats.xlsx
    """
    @staticmethod
    def build_summary(order_by: Optional[str] = None) -> pd.DataFrame:
        ensure_dirs()
        order_by = order_by or getattr(config, "SUMMARY_FIRST_COL", "Win Rate [%]")
        show_summary = bool(getattr(config, "SHOW_RESULTS_SUMMARY", True))

        stats_dir = output_path("stats")
        files = [f for f in os.listdir(stats_dir) if f.lower().endswith(".xlsx")]
        rows = []
        for f in files:
            try:
                df = pd.read_excel(os.path.join(stats_dir, f))
                df["__source__"] = f
                rows.append(df)
            except Exception as e:
                _log(f"[Results][WARN] No se pudo leer {f}: {e}")
        if not rows:
            _log("[Results] No hay archivos en /stats")
            return pd.DataFrame()

        summary = pd.concat(rows, ignore_index=True)
        if order_by in summary.columns:
            ascending = (order_by in minimize_metrics())  # si es de minimizar, orden ascendente
            summary = summary.sort_values(by=order_by, ascending=ascending)
        else:
            _log(f"[Results][WARN] Columna '{order_by}' no está en summary -> sin ordenar")

        if show_summary:
            out = output_path("results/summary_stats.xlsx")
            summary.to_excel(out, index=False)
            _log(f"[Results] Guardado resumen: {out}")

        return summary
