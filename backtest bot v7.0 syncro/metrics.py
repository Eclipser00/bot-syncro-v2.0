# ================================================================
# metrics.py — Cálculo de métricas (estilo backtesting.py) y ranking
# ================================================================
# - Función global minimize_metrics(): indica qué métricas se minimizan
# - Clase MetricsCalculator: calcula métricas de equity + trades replicando
#   las geometrías de backtesting.py/compute_stats (g-mean, vol compuesta, etc.)
# ================================================================

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# LOG ultraverboso (si molesta, comenta el print)
# ---------------------------------------------------------------
def _log(msg: str) -> None:
    print(msg)


# ---------------------------------------------------------------
# Utilidades estilo backtesting.py
# ---------------------------------------------------------------
def geometric_mean(returns: pd.Series) -> float:
    """
    Media geométrica de una serie de retornos porcentuales (en decimales).
    Si hay retornos <= -100%, devuelve 0 (como backtesting.py).
    """
    if not isinstance(returns, (pd.Series, pd.Index, np.ndarray, list)):
        return float('nan')
    ser = pd.Series(returns, dtype=float).fillna(0.0) + 1.0
    if np.any(ser <= 0):
        return 0.0
    # log-sum / n
    return float(np.exp(np.log(ser).sum() / (len(ser) or np.nan)) - 1.0)


def _infer_calendar_and_freq(index: pd.Index) -> Tuple[int, str, bool]:
    """
    Reproduce la heurística de backtesting.py para:
    - annual_trading_days
    - frecuencia de resample para equity -> day_returns
    Devuelve (annual_trading_days, resample_freq, is_datetime_index)
    """
    is_datetime_index = isinstance(index, pd.DatetimeIndex)
    if not is_datetime_index or len(index) == 0:
        return (252, 'D', False)

    # Días por barra aproximados en el índice
    # Mapeo backtesting.py: 7 -> 'W', 31 -> 'ME', 365 -> 'YE', else -> 'D'
    # annual_trading_days:
    #   52 si freq_days==7
    #   12 si freq_days==31
    #   1  si freq_days==365
    #   si no: 365 si hay fines de semana; si no 252
    diffs = index.to_series().diff()
    # valor típico en días (redondeo)
    try:
        avg = pd.to_timedelta(diffs.dropna().median())
        freq_days = int(round(avg / pd.Timedelta(days=1)))
    except Exception:
        freq_days = 1

    have_weekends = False
    try:
        dow = index.dayofweek.to_series()
        # criterio de backtesting.py: > (2/7 * 0.6) ≈ aparecen fines de semana
        have_weekends = dow.between(5, 6).mean() > 2 / 7 * .6
    except Exception:
        pass

    if freq_days == 7:
        annual = 52
        freq = 'W'
    elif freq_days == 31:
        annual = 12
        freq = 'ME'
    elif freq_days == 365:
        annual = 1
        freq = 'YE'
    else:
        annual = 365 if have_weekends else 252
        freq = 'D'

    return (annual, freq, True)


def _vol_ann_compound(day_returns: pd.Series, gmean_day: float, annual: int) -> float:
    """
    Volatilidad anualizada compuesta como en backtesting.py:
    sqrt( (var + (1+g)^2)^annual - (1+g)^(2*annual) )
    donde var = var(day_returns, ddof=int(bool(shape))).
    """
    if day_returns is None or len(day_returns) == 0 or not np.isfinite(gmean_day):
        return float('nan')
    ddof = int(bool(getattr(day_returns, 'shape', ()) ))
    var = float(np.var(day_returns, ddof=ddof))
    base = (1.0 + gmean_day)
    try:
        vol = np.sqrt((var + base ** 2) ** annual - base ** (2 * annual))
    except FloatingPointError:
        vol = float('nan')
    return float(vol)


# ---------------------------------------------------------------
# Calculadora principal
# ---------------------------------------------------------------
class MetricsCalculator:
    """
    Calcula métricas “estilo backtesting.py” a partir de:
      - equity_curve: Serie con índice temporal y valores monetarios.
      - ohlcv: DataFrame original (usado para B&H y como benchmark).
      - trades_df: DataFrame de trades (opcional) para métricas por trade.
      - returns_series: Serie de retornos por barra (opcional, para compatibilidad).
      - periods_per_year: solo para compat. (ya no es necesario).
      - risk_free_rate: en proporción anual (0..1). Default 0.0

    Devuelve un dict listo para df_metrics.
    """
    def __init__(self, risk_free_rate: float = 0.0, warmup_bars: int = 0) -> None:
        assert -1.0 < risk_free_rate < 1.0, "risk_free_rate debe estar en proporción anual"
        self.risk_free_rate = float(risk_free_rate)
        self.warmup_bars = int(max(0, warmup_bars))

    # ------------------------------
    # API pública
    # ------------------------------
    def calculate_all(
        self,
        equity_curve: pd.Series,
        ohlcv: pd.DataFrame,
        trades_df: Optional[pd.DataFrame],
        initial_cash: float,
        returns_series: Optional[pd.Series] = None,
        periods_per_year: Optional[int] = None,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if equity_curve is None or len(equity_curve) == 0:
            _log("[METRICS] Equity vacía: devolviendo NaNs")
            return self._nan_payload()

        # ---------- Drawdown por barra ----------
        equity = pd.Series(equity_curve).astype(float)
        equity.index = pd.DatetimeIndex(equity.index) if not isinstance(equity.index, pd.DatetimeIndex) else equity.index
        run_max = equity.cummax()
        dd = 1.0 - equity / run_max.replace(0, np.nan)
        max_dd = float(-np.nan_to_num(dd.max()))  # negativo

        # ---------- Drawdown promedio y duraciones ----------
        # Avg. Drawdown: promedio de todos los drawdowns (solo cuando dd > 0)
        avg_dd = float(-dd[dd > 0].mean()) if (dd > 0).any() else 0.0

        # Duraciones de drawdown
        dd_durations = []
        in_dd = False
        current_duration = 0

        for dd_val in dd:
            if dd_val > 0:  # Estamos en drawdown
                current_duration += 1
                in_dd = True
            else:  # No hay drawdown
                if in_dd and current_duration > 0:
                    dd_durations.append(current_duration)
                    current_duration = 0
                in_dd = False

        # Si terminamos en drawdown, agregar la última duración
        if in_dd and current_duration > 0:
            dd_durations.append(current_duration)

        max_dd_duration = float(max(dd_durations)) if dd_durations else 0.0
        avg_dd_duration = float(np.mean(dd_durations)) if dd_durations else 0.0

        # ---------- B&H (respetando warmup) ----------
        bh = np.nan
        if ohlcv is not None and 'close' in ohlcv.columns and len(ohlcv) > self.warmup_bars + 1:
            c = ohlcv['close'].values
            bh = float((c[-1] - c[self.warmup_bars]) / c[self.warmup_bars] * 100.0)

        # ---------- Resample + retornos diarios ----------
        annual, freq, is_dt = _infer_calendar_and_freq(equity.index)
        if is_dt:
            day_series = equity.resample(freq).last().dropna()
            day_returns = day_series.pct_change().dropna()
            gmean_day = geometric_mean(day_returns)
            annual_return = (1.0 + gmean_day) ** annual - 1.0
            vol_ann = _vol_ann_compound(day_returns, gmean_day, annual)
            # Sortino (negativos diarios)
            neg = day_returns.clip(upper=0.0)
            try:
                sortino = (annual_return - self.risk_free_rate) / (np.sqrt(np.mean(neg ** 2)) * np.sqrt(annual))
            except ZeroDivisionError:
                sortino = np.nan
        else:
            # Índice no temporal: caer en NaNs controlados
            day_returns = pd.Series(dtype=float)
            gmean_day = float('nan')
            annual_return = float('nan')
            vol_ann = float('nan')
            sortino = float('nan')

        # ---------- Calmar ----------
        calmar = (annual_return / (-max_dd or np.nan)) if np.isfinite(max_dd) else np.nan

        # ---------- Sharpe ----------
        # Igual que backtesting.py: si trabajas en % -> (Return% - rf%)/Vol%
        # Aquí usamos decimales:
        sharpe = (annual_return - self.risk_free_rate) / (vol_ann or np.nan)

        # ---------- Equity Peak ----------
        equity_peak = float(np.nanmax(equity.values)) if len(equity) else np.nan

        # ---------- Ensamblar métricas core ----------
        metrics.update({
            "Equity Peak [$]": equity_peak,
            "Return [%]": float((equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0] * 100.0),
            "Buy & Hold Return [%]": bh,
            "Return (Ann.) [%]": float(annual_return * 100.0),
            "Volatility (Ann.) [%]": float(vol_ann * 100.0),
            "Sharpe Ratio": float(sharpe),
            "Sortino Ratio": float(sortino),
            "Max. Drawdown [%]": float(max_dd * 100.0),
            "Avg. Drawdown [%]": float(avg_dd * 100.0),  
            "Max. Drawdown Duration": max_dd_duration,     
            "Avg. Drawdown Duration": avg_dd_duration,     
                })

        # ---------- Estadísticas de trades ----------
        if trades_df is not None and len(trades_df) > 0:
            tstats = self._trade_stats(trades_df)
            metrics.update(tstats)
        else:
            metrics.update({
                "# Trades": 0,
                "Win Rate [%]": np.nan,
                "Best Trade [%]": np.nan,
                "Worst Trade [%]": np.nan,
                "Avg. Trade [%]": np.nan,
                "Max. Trade Duration": np.nan,
                "Avg. Trade Duration": np.nan,
                "Profit Factor": np.nan,
                "Expectancy [%]": np.nan,
                "SQN": np.nan,
            })

        return metrics

    # ------------------------------
    # Estadísticas de trades
    # ------------------------------
    def _trade_stats(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        df = trades_df.copy()

        # Campos esperados: 'ReturnPct' o ('pnl' y tamaño/base para %).
        # Vamos a soportar ambos nombres (tu pipeline suele tener 'pnl' y quizá 'ReturnPct').
        if 'ReturnPct' in df.columns:
            trade_rets = pd.Series(df['ReturnPct'], dtype=float)
        elif 'return_pct' in df.columns:
            trade_rets = pd.Series(df['return_pct'], dtype=float)
        else:
            # fallback: construir % por trade si existen 'pnl' y 'price_open'
            if {'pnl', 'price_open', 'size'}.issubset(df.columns):
                # aproximación: % sobre precio de entrada * signo del size
                base = df['price_open'].replace(0, np.nan).astype(float)
                base_value = abs(df['size'].astype(float)) * df['price_open'].replace(0, np.nan).astype(float)
                trade_rets = (df['pnl'].astype(float) / base_value.replace(0, np.nan))
            else:
                trade_rets = pd.Series(dtype=float)

        n_trades = int(len(df))
        pl = pd.Series(df.get('pnl', df.get('PnL', np.nan)), dtype=float)
        durations = df.get('barlen', df.get('Duration', np.nan))

        # Win rate
        win_rate = float((pl > 0).mean()) if n_trades else np.nan

        # Best/Worst
        best = float(np.nanmax(trade_rets)) * 100.0 if len(trade_rets) else np.nan
        worst = float(np.nanmin(trade_rets)) * 100.0 if len(trade_rets) else np.nan

        # Avg Trade [%] — geométrica como backtesting.py
        avg_trade = geometric_mean(trade_rets) * 100.0 if len(trade_rets) else np.nan

        # --- Avg.Win / Avg Loss (en % por trade, media aritmética) ---
        wins = trade_rets[trade_rets > 0]
        losses = trade_rets[trade_rets < 0]
        avg_win_pct  = float(wins.mean() * 100.0) if len(wins) else np.nan
        avg_loss_pct = float(losses.mean() * 100.0) if len(losses) else np.nan

        # Duraciones (si están)
        max_dur = durations.max() if hasattr(durations, 'max') else np.nan
        avg_dur = durations.mean() if hasattr(durations, 'mean') else np.nan

        # Profit Factor
        gains = float(pl[pl > 0].sum()) if len(pl) else 0.0
        losses = float(pl[pl < 0].sum()) if len(pl) else 0.0
        pf = gains / (abs(losses) or np.nan)

        # Expectancy [%] — aritmética (como backtesting.py)
        expectancy = float(trade_rets.mean()) * 100.0 if len(trade_rets) else np.nan

        # SQN
        try:
            sqn = float(np.sqrt(n_trades) * pl.mean() / (pl.std() or np.nan)) if n_trades else np.nan
        except ZeroDivisionError:
            sqn = np.nan

        out = {
            "# Trades": n_trades,
            "Win Rate [%]": win_rate * 100.0 if np.isfinite(win_rate) else np.nan,
            "Best Trade [%]": best,
            "Worst Trade [%]": worst,
            "Avg. Trade [%]": avg_trade,
            "Max. Trade Duration": max_dur,
            "Avg. Trade Duration": avg_dur,
            "Avg.Win [%]":  avg_win_pct,
            "Avg.Loss [%]": avg_loss_pct,
            "Profit Factor": pf,
            "Expectancy [%]": expectancy,
            "SQN": sqn,
        }
        _log(f"[METRICS][_trade_stats] {out}")
        return out

    # ------------------------------
    # Payload NaN
    # ------------------------------
    def _nan_payload(self) -> Dict[str, float]:
        return {
            "Equity Peak [$]": np.nan,
            "Return [%]": np.nan,
            "Buy & Hold Return [%]": np.nan,
            "Return (Ann.) [%]": np.nan,
            "Volatility (Ann.) [%]": np.nan,
            "Sharpe Ratio": np.nan,
            "Sortino Ratio": np.nan,
            "Max. Drawdown [%]": np.nan,
            "Avg. Drawdown [%]": np.nan,
            "Max. Drawdown Duration": np.nan,
            "Avg. Drawdown Duration": np.nan,
            "# Trades": 0,
            "Win Rate [%]": np.nan,
            "Best Trade [%]": np.nan,
            "Worst Trade [%]": np.nan,
            "Avg. Trade [%]": np.nan,
            "Max. Trade Duration": np.nan,
            "Avg. Trade Duration": np.nan,
            "Profit Factor": np.nan,
            "Expectancy [%]": np.nan,
            "SQN": np.nan,
        }


# ---------------------------------------------------------------
# Métricas que se minimizan en ranking
# ---------------------------------------------------------------
def minimize_metrics() -> set:
    """
    Devuelve el set de claves de métricas cuyo criterio de ranking es minimizar.
    """
    mins = {
        "Volatility (Ann.) [%]",
        "Max. Drawdown [%]",
        # Métricas de robustez si las añades en el grid:
        "Stress_Sensitivity",
        "WF_iqr_sharpe_oos",
    }
    #_log(f"[METRICS] minimize_metrics -> {sorted(list(mins))}")
    return mins



