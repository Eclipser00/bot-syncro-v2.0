# ================================================================
# overfitting.py — Análisis de robustez (CSCV / Stress / Walk-Forward)
# ================================================================
# - Sin dependencias de functions.py
# - Re-simulación siempre con Backtest.run_backtrader_core (núcleo central)
# - Clases separadas:
#       * CSCV        (alias: cscv)
#       * Stress
#       * WalkForward (alias: Walk_foward)
# - OverfittingManager: punto único de entrada para lanzar los tests
# - Comentarios y logs muy verbosos para depurar paso a paso
# ================================================================

from __future__ import annotations
from typing import Tuple, Dict, Optional, List, Any
import numpy as np
import pandas as pd

from backtest import Backtest  # Usamos el núcleo central de ejecución


# ------------------------------------------------
# Helpers de logging/validación
# ------------------------------------------------
def _log_header(title: str):
    print("\n" + "="*80)
    print(title)
    print("="*80)
# Explicación:
# Este helper imprime un encabezado llamativo en la terminal para separar bloques
# importantes de la ejecución (inicio de CSCV, Stress, WF, etc). Muy útil cuando
# activas muchos logs y necesitas ubicarte visualmente.


def _ensure_flat(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:
        return df.reset_index()
    return df.copy()
# Explicación:
# Algunas operaciones dejan índices con nombre o MultiIndex. Para simplificar
# la manipulación (sort, loc, etc.) normalizamos a un DataFrame “plano”.


def _normalize_params(mat: np.ndarray):
    lo = np.nanmin(mat, axis=0)
    hi = np.nanmax(mat, axis=0)
    rng = np.where((hi - lo) == 0, 1.0, (hi - lo))
    return (mat - lo) / rng
# Explicación:
# Normalización 0..1 por columna para calcular distancias en el espacio de
# parámetros (selección diversificada). Evita que una escala domine a otra.


def _fold_bounds_by_count(ohlcv_index, n_partitions: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Divide el histórico en n_partitions bloques OOS contiguos (leave-1-block-out).
    """
    N = len(ohlcv_index)
    if N < n_partitions:
        raise ValueError(f"Datos insuficientes ({N} barras) para {n_partitions} particiones.")
    edges = np.linspace(0, N, n_partitions + 1, dtype=int)
    bounds = []
    for i in range(n_partitions):
        s = ohlcv_index[edges[i]]
        e = ohlcv_index[edges[i+1]-1] if edges[i+1]-1 < N else ohlcv_index[-1]
        bounds.append((s, e))
    print(f"[FOLDS] Generados {len(bounds)} bloques OOS.")
    return bounds
# Explicación:
# Genera n segmentos consecutivos (folds) que se usarán como OOS. Cada fold
# define un intervalo [start, end] del índice temporal.


def _select_candidates_diversified(pool_sorted: pd.DataFrame, param_cols, metric: str, top_k: int) -> pd.DataFrame:
    P  = pool_sorted[list(param_cols)].to_numpy(dtype=float)
    Pn = _normalize_params(P)
    chosen = [0]  # empezamos por el mejor por métrica
    while len(chosen) < min(top_k, len(pool_sorted)):
        mask = np.ones(len(pool_sorted), dtype=bool)
        mask[chosen] = False
        dmins = []
        for i in np.where(mask)[0]:
            d_to_chosen = np.linalg.norm(Pn[i] - Pn[chosen], axis=1)
            dmins.append((i, float(np.min(d_to_chosen))))
        if not dmins:
            break
        max_d = max(dmins, key=lambda t: t[1])[1]
        candidates = [i for (i, d) in dmins if abs(d - max_d) < 1e-12]
        if len(candidates) > 1:
            best = max(candidates, key=lambda ix: pool_sorted.loc[ix, metric])
            chosen.append(best)
        else:
            chosen.append(candidates[0])
    return pool_sorted.loc[sorted(chosen)].copy()
# Explicación:
# Selección greedy que prioriza diversidad de parámetros. Siempre incluye el
# mejor por métrica y luego añade el más alejado en el espacio normalizado.


# ------------------------------------------------
# Helper interno: ejecutar una sola corrida con tu núcleo
# ------------------------------------------------
def _run_bt_once_with_core(
    df_or_list,
    strategy_cls,
    params: Dict[str, Any],
    base_broker_cfg: Dict[str, Any],
    override_broker_cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Ejecuta una sola corrida de Backtrader sobre un DataFrame o lista de DataFrames.
    Acepta tanto estrategias de un activo (DataFrame único) como multi-activo (List[DataFrame]).
    Usa el método central (Backtest.run_backtrader_core). Permite
    sobreescribir parámetros del broker para tests (comisiones, etc.).

    Devuelve el diccionario de 'stats' de esa corrida.
    """
    # Normalizar: convertir DataFrame único a lista para compatibilidad
    if isinstance(df_or_list, pd.DataFrame):
        df_master_list = [df_or_list]
    elif isinstance(df_or_list, list):
        df_master_list = df_or_list
    else:
        raise ValueError(f"[_run_bt_once_with_core] Tipo no soportado: {type(df_or_list)}")
    
    # Mezclamos config base + overrides (sin mutar el original)
    merged_cfg = dict(base_broker_cfg or {})
    if override_broker_cfg:
        merged_cfg.update({k: override_broker_cfg[k] for k in override_broker_cfg})

    bt_instance = Backtest(broker_cfg=merged_cfg)
    df_trades, df_metrics, df_indicators = bt_instance.run_backtrader_core(
        df_master=df_master_list,     # Lista de DataFrames (1 para single, N para multi-activo)
        strategy_cls=strategy_cls,
        params=params,
        return_trades=False,
        return_metrics=True,
        return_indicators=False
    )
    # Convertir DataFrame de métricas a diccionario
    if df_metrics is not None and not df_metrics.empty:
        # Si df_metrics tiene una sola fila, convertirla a diccionario
        return df_metrics.iloc[0].to_dict()
    return {}
# Explicación:
# Centralizamos la forma de “llamar a Backtrader una vez” para poder reusar
# en CSCV, Stress y WF. No tocamos tu core: solo instanciamos Backtest con
# los parámetros de broker que necesitemos para cada corrida.


# ================================================================
# 1) CSCV + PBO (re-simulación IS/OOS) con Backtrader
# ================================================================
class CSCV:
    """
    Ejecuta Combinatorial Symmetric Cross-Validation con PBO:
      - Selecciona candidatos de un pool top X% por 'metric'
      - Diversifica parámetros (opcional) para evitar clónicos
      - En cada fold: IS = todo salvo el bloque OOS; elegimos ganador IS
      - Evaluamos el ranking de todos OOS y derivamos u, logit(u) y PBO
    """

    def __init__(self, backtest: Backtest):
        self.backtest = backtest

    def run(
        self,
        metrics_df: pd.DataFrame,
        df_master: List[pd.DataFrame],
        strategy_cls,
        param_cols: List[str],
        cfg: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        cfg = cfg or {}
        n_partitions: int   = int(cfg.get("n_partitions", 10))
        top_k: int          = int(cfg.get("top_k", 10))
        metric: str         = str(cfg.get("metric", "Equity Final [$]"))
        selection_mode: str = str(cfg.get("selection_mode", "diversified"))
        top_percent_pool: float = float(cfg.get("top_percent_pool", 0.30))
        broker_override: Optional[Dict[str, Any]] = cfg.get("broker_cfg")  # opcional

        _log_header("CSCV + PBO (Backtrader)")

        # 0) Validaciones mínimas
        if not df_master or len(df_master) == 0 or strategy_cls is None:
            raise ValueError("[CSCV] Se requiere 'df_master' (lista de DataFrames) y 'strategy_cls'.")
        
        # Usar el primer DataFrame para determinar los folds (todos deben estar sincronizados)
        ohlcv = df_master[0] if df_master else None
        if ohlcv is None or ohlcv.empty:
            raise ValueError("[CSCV] El primer DataFrame de df_master está vacío o es None.")
        df_all = _ensure_flat(metrics_df).copy()
        for c in param_cols:
            if c not in df_all.columns:
                raise ValueError(f"[CSCV] Falta columna de parámetro '{c}' en metrics_df.")

        # 1) Métrica de ranking/filtro (fallback razonable)
        if metric not in df_all.columns:
            for m in ['Sharpe Ratio', 'Equity Final [$]', 'Return [%]']:
                if m in df_all.columns:
                    metric = m
                    print(f"[CSCV][WARN] Métrica no encontrada. Uso '{metric}'.")
                    break

        # 2) Construcción de pool top X%
        thr = np.nanpercentile(df_all[metric].values, (1 - top_percent_pool) * 100)
        pool = df_all[df_all[metric] >= thr].copy()
        if len(pool) < top_k:
            pool = df_all.sort_values(metric, ascending=False).head(max(top_k, len(df_all))).copy()
        pool_sorted = pool.sort_values(metric, ascending=False).reset_index(drop=True)

        # 3) Selección de candidatos
        if selection_mode == 'topk':
            cand = pool_sorted.head(top_k).copy()
        elif selection_mode == 'diversified':
            cand = _select_candidates_diversified(pool_sorted, param_cols, metric, top_k)
        else:
            cand = pool_sorted.head(top_k).copy()

        cand = cand.reset_index(drop=True)
        print(f"[CSCV] Pool={len(pool)} | candidatos={len(cand)} | métrica='{metric}'")

        # 4) Particiones
        bounds = _fold_bounds_by_count(ohlcv.index, n_partitions=n_partitions)

        # 5) Re-simulación IS/OOS
        C = len(cand)
        oos_ranks_by_cand = {i: [] for i in range(C)}
        oos_u_by_cand     = {i: [] for i in range(C)}
        selected_is_count = {i: 0 for i in range(C)}
        winner_logit_us   = []

        for fold_idx, (oos_start, oos_end) in enumerate(bounds, start=1):
            print(f"\n[CSCV] Fold {fold_idx}/{len(bounds)} | OOS: {oos_start} → {oos_end}")

            # Dividir TODOS los DataFrames en IS y OOS (multi-activo compatible)
            is_df_list = []
            oos_df_list = []
            
            for df in df_master:
                is_left  = df.loc[(df.index <  oos_start)]
                is_right = df.loc[(df.index >  oos_end)]
                is_df_item = pd.concat([is_left, is_right]).sort_index()
                oos_df_item = df.loc[(df.index >= oos_start) & (df.index <= oos_end)]
                is_df_list.append(is_df_item)
                oos_df_list.append(oos_df_item)

            # Buffer temporal mínimo para que los indicadores “arranquen”
            # Validar usando el primer DataFrame (todos deben tener el mismo índice tras sincronización)
            max_period = max([int(row[k]) for _, row in cand.iterrows() for k in param_cols])
            min_required = max(max_period + 50, 200)
            if len(is_df_list[0]) < min_required or len(oos_df_list[0]) < max_period + 20:
                print(f"[CSCV][WARN] IS/OOS insuficientes (IS={len(is_df_list[0])}, OOS={len(oos_df_list[0])}, min_required={min_required}). Skip.")
                continue

            # IS: elegimos ganador por 'metric'
            is_vals = []
            for _, row in cand.iterrows():
                params = {k: int(row[k]) for k in param_cols}
                met = _run_bt_once_with_core(is_df_list, strategy_cls, params, self.backtest.broker_cfg, broker_override)
                is_vals.append(met.get(metric, np.nan))
            winner_idx = int(np.nanargmax(np.array(is_vals)))
            selected_is_count[winner_idx] += 1
            print(f"[CSCV] Ganador IS -> idx={winner_idx} params={ {k:int(cand.loc[winner_idx,k]) for k in param_cols} } {metric}={is_vals[winner_idx]}")

            # OOS: ranking de todos
            oos_vals = []
            for _, row in cand.iterrows():
                params = {k: int(row[k]) for k in param_cols}
                met = _run_bt_once_with_core(oos_df_list, strategy_cls, params, self.backtest.broker_cfg, broker_override)
                oos_vals.append(met.get(metric, np.nan))

            order = np.argsort(-np.array(oos_vals))
            rank  = np.empty_like(order)
            rank[order] = np.arange(1, len(order) + 1)
            N = len(order)
            u = rank / N

            for i in range(C):
                oos_ranks_by_cand[i].append(int(rank[i]))
                oos_u_by_cand[i].append(float(u[i]))

            u_winner = float(u[winner_idx])
            u_winner = min(max(u_winner, 1e-6), 1 - 1e-6)
            logit_u  = np.log(u_winner / (1.0 - u_winner))
            winner_logit_us.append(logit_u)
            print(f"[CSCV] Rank OOS ganador IS: {int(rank[winner_idx])}/{N} -> u={u_winner:.4f} | logit(u)={logit_u:.4f}")

        # 6) PBO global
        if len(winner_logit_us) == 0:
            print("[CSCV][WARN] Sin folds válidos. PBO=NaN.")
            pbo = np.nan
        else:
            pbo = float(np.mean([1.0 if v < 0 else 0.0 for v in winner_logit_us]))
        print(f"\n[CSCV] PBO estimado (GLOBAL): {pbo:.3f}")

        # 7) Tabla por candidato
        rows = []
        for i, row in cand.reset_index(drop=True).iterrows():
            ranks = oos_ranks_by_cand[i]
            us    = oos_u_by_cand[i]
            wins  = selected_is_count[i]
            out = {
                **{k: int(row[k]) for k in param_cols},
                metric: float(row[metric]),
                'CSCV_PBO': pbo,
                'CSCV_u_mean': float(np.nanmean(us)) if len(us) else np.nan,
                'CSCV_u_median': float(np.nanmedian(us)) if len(us) else np.nan,
                'CSCV_rank_mean': float(np.nanmean(ranks)) if len(ranks) else np.nan,
                'CSCV_rank_median': float(np.nanmedian(ranks)) if len(ranks) else np.nan,
                'CSCV_is_wins': int(wins),
                'CSCV_is_win_rate': float(wins / max(1, len(bounds))),
                'CSCV_Notes': f"selection={selection_mode}, pool={len(pool)}, cand={len(cand)}, folds={len(bounds)}"
            }
            rows.append(out)

        return pd.DataFrame(rows)

    @staticmethod
    def summarize(cscv_df: pd.DataFrame) -> pd.DataFrame:
        _log_header("Resumen de CSCV (1 fila)")
        if cscv_df is None or cscv_df.empty:
            raise ValueError("cscv_df vacío/None.")
        pbo_vals = cscv_df['CSCV_PBO'].dropna().unique()
        pbo = float(pbo_vals[0]) if len(pbo_vals) else np.nan
        u_med  = float(np.nanmedian(cscv_df.get('CSCV_u_median', pd.Series(dtype=float))))
        rk_med = float(np.nanmedian(cscv_df.get('CSCV_rank_median', pd.Series(dtype=float))))
        print(f"[CSCV] PBO={pbo:.3f} | u_med={u_med:.3f} | rk_med={rk_med:.3f}")
        return pd.DataFrame({'CSCV_PBO':[pbo], 'CSCV_u_median_med':[u_med], 'CSCV_rank_median_med':[rk_med]})
# Explicación:
# Esta clase empaqueta todo el flujo CSCV adaptado a tu núcleo. La selección
# “diversified” replica tu versión anterior. La función summarize produce
# un resumen 1xN con PBO y estadísticos principales.


# ================================================================
# 2) STRESS de Costes (comisiones/slippage) sobre GRID
# ================================================================
class Stress:
    """
    Evalúa la sensibilidad del desempeño (p. ej., Equity Final) frente al coste
    operativo (comisión y slippage). Ajusta la comisión desde cost_grid y, por
    limitaciones del core actual, el slippage_bps se IGNORA (ver comentario).
    """

    def __init__(self, backtest: Backtest):
        self.backtest = backtest

    def run(
        self,
        ohlcv: pd.DataFrame,
        strategy_cls,
        grid_df: pd.DataFrame,
        param_cols: List[str],
        cost_grid: Dict[str, List[Any]],
        cfg: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        cfg = cfg or {}
        rank_metric: str     = str(cfg.get("rank_metric", "Sharpe Ratio"))
        top_percent_pool: float = float(cfg.get("top_percent_pool", 0.30))
        max_candidates: int  = int(cfg.get("max_candidates", 24))
        selection_mode: str  = str(cfg.get("selection_mode", "diversified"))
        broker_override: Optional[Dict[str, Any]] = cfg.get("broker_cfg")  # opcional

        _log_header("Stress de Costes EN REJILLA")

        # 0) Validaciones
        if ohlcv is None or strategy_cls is None:
            raise ValueError("[STRESS] Se requiere 'ohlcv' y 'strategy_cls'.")
        df = _ensure_flat(grid_df).copy()
        for p in param_cols:
            if p not in df.columns:
                raise ValueError(f"[STRESS] Falta parámetro '{p}' en grid_df.")

        if rank_metric not in df.columns:
            for m in ['Sharpe Ratio','Equity Final [$]','Return [%]']:
                if m in df.columns:
                    rank_metric = m
                    print(f"[STRESS][WARN] rank_metric no estaba; uso '{rank_metric}'.")
                    break

        # Pool
        thr = np.nanpercentile(df[rank_metric].values, (1 - top_percent_pool)*100)
        pool = df[df[rank_metric] >= thr].copy()
        if len(pool) < max_candidates:
            pool = df.sort_values(rank_metric, ascending=False).head(max_candidates).copy()
        pool_sorted = pool.sort_values(rank_metric, ascending=False).reset_index(drop=True)

        # Selección de candidatos
        if selection_mode == 'diversified':
            cand = _select_candidates_diversified(pool_sorted, param_cols, rank_metric, max_candidates)
        else:
            cand = pool_sorted.head(max_candidates).copy()
        cand = cand.reset_index(drop=True)

        print(f"[STRESS] Pool={len(pool)} | candidatos={len(cand)} | métrica='{rank_metric}'")

        # Cost grid
        commissions = list(cost_grid.get('commission', [self.backtest.broker_cfg.get('commission', 0.0)]))
        slipp_bps  = list(cost_grid.get('slippage_bps', [0]))
        if any(x != 0 for x in slipp_bps):
            print("[STRESS][INFO] slippage_bps presente pero el core actual no soporta slippage; se IGNORA (solo comisión).")
        # NOTA IMPORTANTE:
        # Tu run_backtrader_core no aplica slippage (y no podemos modificarlo sin
        # romper tu requisito). Por eso aquí variamos únicamente la comisión.

        results = []
        for idx, row in cand.iterrows():
            params = {k: int(row[k]) for k in param_cols}
            print(f"[STRESS] #{idx+1}/{len(cand)} params={params}")
            xy = []
            for c in commissions:
                override = dict(broker_override or {})
                override['commission'] = float(c)
                met = _run_bt_once_with_core(ohlcv, strategy_cls, params, self.backtest.broker_cfg, override)
                equity = float(met.get('Equity Final [$]', np.nan))
                # “Coste total” reportado: comisión + slippage_bps/10000 (aunque slippage no aplique en core)
                # Se mantiene para continuidad con tu versión previa.
                for bps in slipp_bps:
                    cost_total = float(c) + (int(bps) / 10000.0)
                    xy.append((cost_total, equity))
                    print(f"   [STRESS] commission={c} slippage_bps={bps} (ignored) -> Equity={equity}")

            x = np.array([t[0] for t in xy], dtype=float)
            y = np.array([t[1] for t in xy], dtype=float)
            if np.isfinite(y).sum() >= 2 and len(np.unique(x)) >= 2:
                slope = np.polyfit(x, y, deg=1)[0]
                stress_score = -slope
            else:
                slope = np.nan
                stress_score = np.nan
                print("   [STRESS][WARN] Datos insuficientes para slope en este candidato.")

            results.append({
                **params,
                rank_metric: float(row[rank_metric]),
                'Stress_Sensitivity': float(stress_score) if pd.notna(stress_score) else np.nan,
                'Stress_Slope_Equity_vs_Costs': float(slope) if pd.notna(slope) else np.nan,
                'Stress_Notes': f"pool={len(pool)}, cand={len(cand)}, points={len(xy)}"
            })

        out = pd.DataFrame(results)
        print(f"[STRESS] Completado. Candidatos evaluados: {len(out)}")
        return out

    @staticmethod
    def summarize(stress_grid_df: pd.DataFrame, how: str = 'median') -> pd.DataFrame:
        _log_header("Resumen de Stress-GRID (1 fila)")
        if stress_grid_df is None or stress_grid_df.empty:
            raise ValueError("stress_grid_df vacío/None.")

        s = stress_grid_df['Stress_Sensitivity'].astype(float)
        p25 = float(np.nanpercentile(s, 25)) if s.notna().any() else np.nan
        p50 = float(np.nanmedian(s)) if s.notna().any() else np.nan
        p75 = float(np.nanpercentile(s, 75)) if s.notna().any() else np.nan

        chosen = p50 if how == 'median' else (p75 if how == 'p75' else p25)
        print(f"[STRESS] p25={p25:.4f} | p50={p50:.4f} | p75={p75:.4f} | usado='{how}' -> {chosen:.4f}")

        return pd.DataFrame({
            'Stress_Sensitivity': [chosen],
            'Stress_Sens_p25': [p25],
            'Stress_Sens_p50': [p50],
            'Stress_Sens_p75': [p75],
            'Stress_Notes': [f"resumen '{how}' sobre {len(stress_grid_df)} candidatos"]
        })
# Explicación:
# La clase reusa tu idea de slope (pendiente equity vs coste). Mientras el core
# no soporte slippage, lo dejamos explícitamente ignorado para no tocar tu
# método central. En cuanto habilitemos slippage en BacktraderCore, este módulo
# ya está listo para usarlo (override_broker_cfg).


# ================================================================
# 3) Walk-Forward (GRID de candidatos)
# ================================================================
class WalkForward:
    """
    Evaluación Walk-Forward sobre candidatos del grid:
      - Divide el histórico en n_folds OOS
      - Evalúa la métrica OOS por fold y resume con mediana e IQR
      - Opcional: normalización 0..1 para combinar medianas y IQR en un score
    """

    def __init__(self, backtest: Backtest):
        self.backtest = backtest

    def run(
        self,
        ohlcv: pd.DataFrame,
        strategy_cls,
        grid_df: pd.DataFrame,
        param_cols: List[str],
        cfg: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        cfg = cfg or {}
        wf_config: Dict[str, Any] = cfg.get("wf_config", {"n_folds": 5, "scheme": "rolling"})
        metric: str          = str(cfg.get("metric", "Sharpe Ratio"))
        top_percent_pool: float = float(cfg.get("top_percent_pool", 0.30))
        max_candidates: int  = int(cfg.get("max_candidates", 24))
        selection_mode: str  = str(cfg.get("selection_mode", "diversified"))
        broker_override: Optional[Dict[str, Any]] = cfg.get("broker_cfg")
        normalize_to_01: bool = bool(cfg.get("normalize_to_01", True))
        norm_quantiles: Tuple[int, int] = tuple(cfg.get("norm_quantiles", (5, 95)))

        _log_header("WF GRID (Backtrader)")

        if ohlcv is None or strategy_cls is None:
            raise ValueError("[WF] Se requiere 'ohlcv' y 'strategy_cls'.")

        df = _ensure_flat(grid_df).copy()
        if metric not in df.columns:
            for m in ['Sharpe Ratio','Equity Final [$]','Return [%]']:
                if m in df.columns:
                    metric = m
                    print(f"[WF][WARN] Métrica no encontrada. Uso '{metric}'.")
                    break

        for p in param_cols:
            if p not in df.columns:
                raise ValueError(f"[WF] Falta parámetro '{p}' en grid_df.")

        thr = np.nanpercentile(df[metric].values, (1 - top_percent_pool)*100)
        pool = df[df[metric] >= thr].copy()
        if len(pool) < max_candidates:
            pool = df.sort_values(metric, ascending=False).head(max_candidates).copy()
        pool_sorted = pool.sort_values(metric, ascending=False).reset_index(drop=True)

        if selection_mode == 'diversified':
            cand = _select_candidates_diversified(pool_sorted, param_cols, metric, max_candidates)
        else:
            cand = pool_sorted.head(max_candidates).copy()
        cand = cand.reset_index(drop=True)

        print(f"[WF] Pool={len(pool)} | candidatos={len(cand)} | métrica='{metric}'")

        # Folds OOS
        n_folds = int(wf_config.get('n_folds', 5))
        bounds = _fold_bounds_by_count(ohlcv.index, n_partitions=n_folds)

        results = []
        for i, row in cand.iterrows():
            params = {k: int(row[k]) for k in param_cols}
            print(f"[WF] #{i+1}/{len(cand)} params={params}")
            oos_vals = []
            for j, (oos_start, oos_end) in enumerate(bounds, start=1):
                oos_df = ohlcv.loc[(ohlcv.index >= oos_start) & (ohlcv.index <= oos_end)]
                if len(oos_df) < 20:
                    print(f"   [WF] Fold {j}: OOS corto -> skip")
                    continue
                met = _run_bt_once_with_core(oos_df, strategy_cls, params, self.backtest.broker_cfg, broker_override)
                val = met.get(metric, np.nan)
                if pd.isna(val):
                    val = met.get('Equity Final [$]', np.nan)
                oos_vals.append(val)
                print(f"   [WF] Fold {j}: OOS {metric}={val}")

            med = float(np.nanmedian(oos_vals)) if oos_vals else np.nan
            q1  = float(np.nanpercentile(oos_vals, 25)) if oos_vals else np.nan
            q3  = float(np.nanpercentile(oos_vals, 75)) if oos_vals else np.nan
            iqr = float(q3 - q1) if (not np.isnan(q3) and not np.isnan(q1)) else np.nan

            results.append({
                **params,
                'WF_OOS_median': med,
                'WF_OOS_iqr': iqr,
                'WF_folds': n_folds
            })

        wf_df = pd.DataFrame(results)
        print(f"[WF] Completado. Combinaciones evaluadas: {len(wf_df)}")

        # Normalización 0..1 opcional
        if normalize_to_01 and not wf_df.empty:
            def _norm01(series: pd.Series, q=(5, 95), invert=False) -> pd.Series:
                s = series.astype(float).copy()
                if s.notna().sum() < 2:
                    return pd.Series([0.5] * len(s), index=s.index)
                lo, hi = np.nanpercentile(s, q[0]), np.nanpercentile(s, q[1])
                rng = max(1e-12, hi - lo)
                norm = (s - lo) / rng
                norm = norm.clip(0.0, 1.0)
                return 1.0 - norm if invert else norm

            wf_df['WF_OOS_median_01'] = _norm01(wf_df['WF_OOS_median'], q=norm_quantiles, invert=False)
            wf_df['WF_OOS_iqr_01']    = _norm01(wf_df['WF_OOS_iqr'],    q=norm_quantiles, invert=True)
            wf_df['WF_OOS_score_01']  = 0.5 * wf_df['WF_OOS_median_01'] + 0.5 * wf_df['WF_OOS_iqr_01']
            print(f"[WF] Normalización 0..1 aplicada (q{norm_quantiles}).")

        return wf_df

    @staticmethod
    def summarize(wf_df: pd.DataFrame) -> pd.DataFrame:
        _log_header("Resumen de WF-GRID (1 fila)")
        if wf_df is None or wf_df.empty:
            raise ValueError("wf_df vacío/None.")

        if all(col in wf_df.columns for col in ['WF_OOS_median_01','WF_OOS_iqr_01','WF_OOS_score_01']):
            med_p50 = float(np.nanmedian(wf_df['WF_OOS_median_01'].astype(float)))
            iqr_p50 = float(np.nanmedian(wf_df['WF_OOS_iqr_01'].astype(float)))
            sc_p50  = float(np.nanmedian(wf_df['WF_OOS_score_01'].astype(float)))
            print(f"[WF] (0..1) median_p50={med_p50:.3f} | iqr_p50={iqr_p50:.3f} | score_p50={sc_p50:.3f}")
            return pd.DataFrame({
                'WF_median_oos_01': [med_p50],
                'WF_iqr_oos_01': [iqr_p50],
                'WF_score_01': [sc_p50],
                'WF_notes': ["Resumen p50 de mediana/IQR/score OOS (normalizado 0..1)"]
            })
        else:
            med = wf_df['WF_OOS_median'].astype(float)
            iqr = wf_df['WF_OOS_iqr'].astype(float)
            med_p50 = float(np.nanmedian(med)) if med.notna().any() else np.nan
            iqr_p50 = float(np.nanmedian(iqr)) if iqr.notna().any() else np.nan
            print(f"[WF] (RAW) median_p50={med_p50:.4f} | iqr_p50={iqr_p50:.4f}")
            return pd.DataFrame({
                'WF_median_sharpe_oos': [med_p50],
                'WF_iqr_sharpe_oos': [iqr_p50],
                'WF_notes': ["Resumen p50 de median/IQR OOS (RAW)"]
            })
# Explicación:
# Implementa el esquema clásico de walk-forward con folds por conteo. La
# normalización opcional permite comparar/combinar medianas con IQR en una
# escala común (0..1), útil para seleccionar candidatos “estables”.


# ================================================================
# OverfittingManager — orquestador de robustez
# ================================================================
class OverfittingManager:
    """
    Punto único para lanzar análisis de robustez:
      - CSCV + PBO
      - Stress de costes
      - Walk-Forward
    Integra con el núcleo Backtest.run_backtrader_core y devuelve tablas y resúmenes.
    """

    def __init__(self, backtest: Backtest):
        self.backtest = backtest

    def evaluate_robustness(
        self,
        strategy_cls,
        param_cols: List[str],
        ohlcv_single: pd.DataFrame,
        n_partitions: int = 10,
        cfg: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta (según cfg) CSCV / Stress / WF y devuelve un diccionario con:
          {
            'cscv':   {'df': DataFrame, 'summary': DataFrame} | None,
            'stress': {'df': DataFrame, 'summary': DataFrame} | None,
            'wf':     {'df': DataFrame, 'summary': DataFrame} | None
          }

        Parámetros clave esperados en cfg:
          - grid_df: DataFrame con el grid (obligatorio para Stress/WF/CSCV).
          - rank_metric: str (para stress/wf cuando aplica).
          - cscv: { 'do': bool, 'top_k': int, 'metric': str, 'selection_mode': 'diversified'|'topk',
                    'top_percent_pool': float, 'broker_cfg': dict(opcional)}
          - stress: { 'do': bool, 'top_percent_pool': float, 'max_candidates': int,
                      'selection_mode': str, 'cost_grid': {'commission': [...], 'slippage_bps': [...]},
                      'rank_metric': str, 'broker_cfg': dict(opcional) }
          - wf: { 'do': bool, 'metric': str, 'top_percent_pool': float, 'max_candidates': int,
                  'selection_mode': str, 'wf_config': {'n_folds': int}, 'broker_cfg': dict(opcional),
                  'normalize_to_01': bool, 'norm_quantiles': (lo, hi) }
        """
        cfg = cfg or {}
        grid_df = cfg.get("grid_df", None)

        out = {'cscv': None, 'stress': None, 'wf': None}

        # 1) CSCV
        c_cfg = dict(cfg.get('cscv', {}))
        if c_cfg.get('do', False):
            # aseguremos n_partitions coherente
            c_cfg.setdefault('n_partitions', n_partitions)
            cscv_runner = CSCV(self.backtest)
            cscv_df = cscv_runner.run(
                metrics_df=grid_df,
                ohlcv=ohlcv_single,
                strategy_cls=strategy_cls,
                param_cols=param_cols,
                cfg=c_cfg
            )
            cscv_sum = CSCV.summarize(cscv_df)
            out['cscv'] = {'df': cscv_df, 'summary': cscv_sum}

        # 2) Stress
        s_cfg = dict(cfg.get('stress', {}))
        if s_cfg.get('do', False):
            cost_grid = s_cfg.get('cost_grid', {'commission': [self.backtest.broker_cfg.get('commission', 0.0)], 'slippage_bps': [0]})
            stress_runner = Stress(self.backtest)
            stress_df = stress_runner.run(
                ohlcv=ohlcv_single,
                strategy_cls=strategy_cls,
                grid_df=grid_df,
                param_cols=param_cols,
                cost_grid=cost_grid,
                cfg=s_cfg
            )
            stress_sum = Stress.summarize(stress_df, how=s_cfg.get('summary_how', 'median'))
            out['stress'] = {'df': stress_df, 'summary': stress_sum}

        # 3) Walk-Forward
        w_cfg = dict(cfg.get('wf', {}))
        if w_cfg.get('do', False):
            wf_runner = WalkForward(self.backtest)
            wf_df = wf_runner.run(
                ohlcv=ohlcv_single,
                strategy_cls=strategy_cls,
                grid_df=grid_df,
                param_cols=param_cols,
                cfg=w_cfg
            )
            wf_sum = WalkForward.summarize(wf_df)
            out['wf'] = {'df': wf_df, 'summary': wf_sum}

        return out
# Explicación:
# Este manager centraliza la ejecución de los tests y encapsula la configuración
# de cada uno. Devuelve tanto las tablas detalladas como resúmenes 1xN para que
# puedas guardar o mostrar resultados rápidamente.


