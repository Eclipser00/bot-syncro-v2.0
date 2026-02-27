"""
Runner mínimo de paridad (Backtrader) para comparar contra el live/paper bot.

Objetivo:
  - Ejecutar PivotZoneTest sobre los mismos CSVs que usa el bot (M1),
    resampleando internamente a M3 (TF_zone) para asemejar el pipeline del bot.
  - Generar `outputs/backtest_events.jsonl` (en el repo root) con `run_id`.

Uso (desde el repo root):
  python "backtest bot v6.1/parity_runner.py" --symbols EURUSD GBPUSD USDJPY

Variables de entorno:
  RUN_ID:            identificador del run (se escribe en cada evento)
  EVENT_LOG_TRUNCATE: si "1", trunca el JSONL al inicializar el logger (solo primer símbolo)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import backtrader as bt
import pandas as pd

import config as bt_config
from backtest import Backtest
from strategies import PivotZoneTest


def _symbol_key(name: str) -> str:
    base = (name or "").split(",")[0].strip()
    if "_" in base:
        base = base.split("_")[-1]
    return base.strip().upper()


def _resolve_csv_path(csv_dir: Path, symbol: str) -> Path:
    symbol_clean = _symbol_key(symbol)
    candidates = [
        csv_dir / f"{symbol}.csv",
        csv_dir / f"{symbol_clean}.csv",
        csv_dir / f"{symbol_clean},.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    wildcard_matches = sorted(csv_dir.glob(f"{symbol_clean}*.csv"))
    if wildcard_matches:
        return wildcard_matches[0]
    raise FileNotFoundError(csv_dir / f"{symbol}.csv")


def _load_bot_params(bot_config_path: Path) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    if not bot_config_path.exists():
        return ({}, {})
    try:
        spec = importlib.util.spec_from_file_location("bot_config_parity", bot_config_path)
        if spec is None or spec.loader is None:
            return ({}, {})
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        settings = getattr(module, "settings", None)
        if settings is None:
            return ({}, {})

        base_params: dict[str, float] = {}
        strat_params = None
        for strat in getattr(settings, "strategies", []):
            if getattr(strat, "name", "") == "PivotZoneTest":
                strat_params = strat
                break
        if strat_params is None and getattr(settings, "strategies", None):
            strat_params = settings.strategies[0]
        if strat_params is not None:
            base_params = {
                "n1": int(getattr(strat_params, "n1", 14)),
                "n2": int(getattr(strat_params, "n2", 60)),
                "n3": int(getattr(strat_params, "n3", 3)),
                "size_pct": float(getattr(strat_params, "size_pct", 0.05)),
                "p": float(getattr(strat_params, "p", 0.50)),
            }

        params_by_symbol: dict[str, dict[str, float]] = {}
        for sym in getattr(settings, "symbols", []):
            key = _symbol_key(getattr(sym, "name", ""))
            overrides: dict[str, float] = {}
            for field in ("n1", "n2", "n3", "size_pct", "p"):
                value = getattr(sym, field, None)
                if value is None:
                    continue
                overrides[field] = int(value) if field in {"n1", "n2", "n3"} else float(value)
            merged = dict(base_params)
            merged.update(overrides)
            if merged:
                params_by_symbol[key] = merged

        return params_by_symbol, base_params
    except Exception as exc:
        print(f"[PARITY][WARN] No se pudo cargar config del bot: {exc}")
        return ({}, {})


def _load_csv_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    if "time" in cols:
        ts = df[cols["time"]]
        unit = "ms" if float(ts.iloc[0]) > 1e12 else "s"
        dt = pd.to_datetime(ts, unit=unit, utc=True)
    elif "datetime" in cols:
        dt = pd.to_datetime(df[cols["datetime"]], utc=True)
    else:
        raise ValueError(f"CSV {path} debe tener columna time o datetime")

    open_arr = pd.to_numeric(df[cols["open"]], errors="coerce").to_numpy()
    high_arr = pd.to_numeric(df[cols["high"]], errors="coerce").to_numpy()
    low_arr = pd.to_numeric(df[cols["low"]], errors="coerce").to_numpy()
    close_arr = pd.to_numeric(df[cols["close"]], errors="coerce").to_numpy()
    if "volume" in cols or "vol" in cols:
        vol_key = cols.get("volume") or cols.get("vol")  # type: ignore[assignment]
        vol_arr = pd.to_numeric(df[vol_key], errors="coerce").to_numpy()  # type: ignore[arg-type]
    else:
        vol_arr = pd.Series([0.0] * len(df)).to_numpy()

    out = pd.DataFrame(
        {"open": open_arr, "high": high_arr, "low": low_arr, "close": close_arr, "volume": vol_arr},
        index=pd.DatetimeIndex(dt).tz_convert(None),  # Backtrader suele trabajar con datetimes naive
    )

    out = out.dropna(subset=["open", "high", "low", "close"]).sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]
    return out


def _configure_broker(cerebro: bt.Cerebro) -> None:
    cash0 = float(getattr(bt_config, "CASH", 100000.0))
    commission_pct = float(getattr(bt_config, "COMMISSION", 0.0)) / 100.0
    margin_factor = 1.0 / max(1.0, float(getattr(bt_config, "MARGIN", 1)))
    trade_on_close = bool(getattr(bt_config, "TRADE_ON_CLOSE", True))

    cerebro.broker.setcash(cash0)
    try:
        cerebro.broker.set_coc(trade_on_close)
    except Exception:
        pass

    comminfo = Backtest._make_commission_info(commission_pct=commission_pct, margin_factor=margin_factor)
    cerebro.broker.addcommissioninfo(comminfo)


def _strategy_params(
    symbol: str,
    bot_params_by_symbol: dict[str, dict[str, float]] | None = None,
    base_params: dict[str, float] | None = None,
) -> dict:
    n1 = int(getattr(bt_config, "N1_RANGE", [14])[0])
    n2 = int(getattr(bt_config, "N2_RANGE", [60])[0])
    n3 = int(getattr(bt_config, "N3_RANGE", [3])[0])
    # size_pct y p: no están en config.py del backtest; usamos defaults de la estrategia
    params: dict[str, float] = {
        "n1": float(n1),
        "n2": float(n2),
        "n3": float(n3),
        "size_pct": 0.05,
        "p": 0.50,
    }
    if base_params:
        params.update(base_params)
    if bot_params_by_symbol:
        key = _symbol_key(symbol)
        if key in bot_params_by_symbol:
            params.update(bot_params_by_symbol[key])
    return params


def _resample_m3(df_m1: pd.DataFrame, drop_last_partial: bool = False) -> pd.DataFrame:
    resampled = (
        # Semántica estándar: etiqueta de cierre (right) para evitar look-ahead en TF_zone.
        df_m1.resample("3min", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    if drop_last_partial and len(resampled) > 0:
        try:
            deltas = df_m1.index.to_series().diff().dropna()
            base_minutes = deltas.dt.total_seconds().median() / 60.0
            if base_minutes and base_minutes > 0:
                expected = max(1, int(round(3.0 / float(base_minutes))))
                last_label = resampled.index[-1]
                bin_start = last_label - pd.Timedelta(minutes=3.0)
                count = df_m1[(df_m1.index > bin_start) & (df_m1.index <= last_label)].shape[0]
                if count < expected:
                    resampled = resampled.iloc[:-1]
            else:
                resampled = resampled.iloc[:-1]
        except Exception:
            resampled = resampled.iloc[:-1]
    return resampled


def run_symbol(
    symbol: str,
    csv_dir: Path,
    zone_compression: int,
    max_bars: int | None,
    bot_params_by_symbol: dict[str, dict[str, float]] | None = None,
    base_params: dict[str, float] | None = None,
) -> None:
    csv_path = _resolve_csv_path(csv_dir, symbol)

    df = _load_csv_ohlcv(csv_path)
    if max_bars is not None and max_bars > 0:
        df = df.iloc[:max_bars].copy()

    cerebro = bt.Cerebro(stdstats=False)
    _configure_broker(cerebro)

    if int(zone_compression) != 3:
        raise ValueError("Este parity_runner solo soporta zone_compression=3 (M3) por ahora.")

    df_m3 = _resample_m3(df, drop_last_partial=True)

    data_m1 = bt.feeds.PandasData(
        dataname=df,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        datetime=None,
        timeframe=bt.TimeFrame.Minutes,
    )
    data_m1._name = f"{symbol}_M1"
    data_m1._symbol = symbol
    data_m1._timeframe_label = "M1"
    cerebro.adddata(data_m1)

    data_m3 = bt.feeds.PandasData(
        dataname=df_m3,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        datetime=None,
        timeframe=bt.TimeFrame.Minutes,
    )
    data_m3._name = f"{symbol}_M3"
    data_m3._symbol = symbol
    data_m3._timeframe_label = "M3"
    cerebro.adddata(data_m3)

    strategy_params = _strategy_params(symbol, bot_params_by_symbol, base_params)
    print(
        f"[PARAMS] {symbol} "
        f"n1={strategy_params.get('n1')} "
        f"n2={strategy_params.get('n2')} "
        f"n3={strategy_params.get('n3')} "
        f"size_pct={strategy_params.get('size_pct')} "
        f"p={strategy_params.get('p')}"
    )
    cerebro.addstrategy(PivotZoneTest, **strategy_params)
    results = cerebro.run(runonce=False, preload=True)
    try:
        strat = results[0]
        zones = getattr(strat, "_saved_zones", None)
        zones_count = len(zones) if isinstance(zones, list) else -1
        bars_m1 = len(df)
        bars_m3 = len(df_m3)
        print(f"[SUMMARY] {symbol} bars_m1={bars_m1} bars_m3={bars_m3} zones_saved={zones_count}")
        if zones_count > 0:
            last = zones[-1]
            print(
                f"[SUMMARY] {symbol} last_zone top={last.get('top')} bot={last.get('bot')} bar={last.get('bar')}"
            )
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Runner de paridad Backtrader (PivotZoneTest).")
    parser.add_argument("--csv_dir", default="last_trading_bot_v1.1_pivot_zone/data_development")
    parser.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "USDJPY"])
    parser.add_argument("--zone_compression", type=int, default=3, help="Compresión para TF_zone (3 => M3).")
    parser.add_argument("--max_bars", type=int, default=0, help="Si >0, recorta cada CSV a las primeras N barras.")
    parser.add_argument(
        "--bot_config",
        default="",
        help="Ruta opcional a config.py del bot para alinear parametros por simbolo.",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    if not csv_dir.exists():
        raise FileNotFoundError(csv_dir)

    bot_config_path = (
        Path(args.bot_config)
        if args.bot_config
        else Path(__file__).resolve().parents[1] / "last_trading_bot_v1.1_pivot_zone" / "config.py"
    )
    bot_params_by_symbol, base_params = _load_bot_params(bot_config_path)
    if bot_params_by_symbol:
        print(f"[PARITY] Parametros del bot cargados desde {bot_config_path}")
    else:
        print("[PARITY] Usando parametros del backtest (no se cargo config del bot).")

    # Truncar solo una vez (primer símbolo) si el caller lo pide
    truncate = os.environ.get("EVENT_LOG_TRUNCATE") == "1"
    max_bars = int(args.max_bars) if int(args.max_bars) > 0 else None

    for i, sym in enumerate(args.symbols):
        if truncate and i > 0:
            os.environ["EVENT_LOG_TRUNCATE"] = "0"
        run_symbol(
            sym,
            csv_dir=csv_dir,
            zone_compression=args.zone_compression,
            max_bars=max_bars,
            bot_params_by_symbol=bot_params_by_symbol,
            base_params=base_params,
        )


if __name__ == "__main__":
    main()
