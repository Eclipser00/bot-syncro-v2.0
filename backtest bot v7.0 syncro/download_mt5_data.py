"""Download OHLCV data from MetaTrader5 and export CSVs into data01.

This script uses in-file toggles (no CLI) and writes CSV files with the same
format used by this backtest project:
    time,open,high,low,close,Volume

`time` is UNIX epoch in seconds (UTC).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import MetaTrader5 as mt5
import pandas as pd


# ============================================================
# TOGGLES (edit these values before running)
# ============================================================
START_UTC = "2026-02-01 00:00:00"  # YYYY-MM-DD HH:MM:SS (UTC)
END_UTC = "2026-02-8 00:00:00"  # YYYY-MM-DD HH:MM:SS (UTC)
TICKERS = ["EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAME = "M1"  # M1|M5|M15|M30|H1|H4|D1|W1|MN1
OUTPUT_DIR = "data01"
OVERWRITE = True


TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


def log(message: str) -> None:
    print(f"[MT5-DOWNLOAD] {message}", flush=True)


def parse_utc_datetime(raw_value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.strptime(raw_value, "%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise ValueError(
            f"{field_name} invalido '{raw_value}'. Usa formato YYYY-MM-DD HH:MM:SS en UTC."
        ) from exc
    return parsed.replace(tzinfo=timezone.utc)


def normalize_tickers(raw_tickers: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_tickers:
        symbol = str(item).strip().upper()
        if not symbol or symbol in seen:
            continue
        normalized.append(symbol)
        seen.add(symbol)
    return normalized


def validate_inputs() -> tuple[datetime, datetime, str, list[str]]:
    start_utc = parse_utc_datetime(START_UTC, "START_UTC")
    end_utc = parse_utc_datetime(END_UTC, "END_UTC")

    if start_utc >= end_utc:
        raise ValueError("START_UTC debe ser menor que END_UTC.")

    timeframe = TIMEFRAME.strip().upper()
    if timeframe not in TIMEFRAME_MAP:
        valid = ", ".join(TIMEFRAME_MAP.keys())
        raise ValueError(f"TIMEFRAME '{TIMEFRAME}' no valido. Debe ser: {valid}.")

    tickers = normalize_tickers(TICKERS)
    if not tickers:
        raise ValueError("TICKERS no puede estar vacio.")

    return start_utc, end_utc, timeframe, tickers


def ensure_symbol(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        if not mt5.symbol_select(symbol, True):
            error_code, error_msg = mt5.last_error()
            raise RuntimeError(
                f"No se pudo habilitar simbolo {symbol}. Error {error_code}: {error_msg}"
            )
        info = mt5.symbol_info(symbol)

    if info is None:
        raise RuntimeError(f"Simbolo {symbol} no disponible en MetaTrader5.")

    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            error_code, error_msg = mt5.last_error()
            raise RuntimeError(
                f"Simbolo {symbol} no visible y no se pudo seleccionar. "
                f"Error {error_code}: {error_msg}"
            )


def fetch_symbol_ohlcv(
    symbol: str,
    mt5_timeframe: int,
    start_utc: datetime,
    end_utc: datetime,
) -> pd.DataFrame:
    ensure_symbol(symbol)
    rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_utc, end_utc)

    if rates is None:
        error_code, error_msg = mt5.last_error()
        raise RuntimeError(
            f"copy_rates_range devolvio None para {symbol}. "
            f"Error {error_code}: {error_msg}"
        )

    if len(rates) == 0:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "Volume"])

    df = pd.DataFrame(rates)
    required_cols = ["time", "open", "high", "low", "close", "tick_volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise RuntimeError(f"Columnas faltantes en respuesta MT5 para {symbol}: {missing}")

    out = df[["time", "open", "high", "low", "close", "tick_volume"]].copy()
    out.rename(columns={"tick_volume": "Volume"}, inplace=True)

    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce")

    out = out.dropna(subset=["time", "open", "high", "low", "close"])
    out = out.sort_values("time").drop_duplicates(subset=["time"], keep="last")

    if out.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "Volume"])

    out["time"] = out["time"].astype("int64")
    out["Volume"] = out["Volume"].fillna(0).astype("int64")

    return out[["time", "open", "high", "low", "close", "Volume"]]


def write_symbol_csv(df: pd.DataFrame, output_path: Path) -> None:
    if output_path.exists() and not OVERWRITE:
        raise FileExistsError(
            f"El archivo {output_path} ya existe y OVERWRITE=False."
        )
    df.to_csv(output_path, index=False)


def main() -> int:
    try:
        start_utc, end_utc, timeframe, tickers = validate_inputs()
    except ValueError as exc:
        log(f"ERROR de validacion: {exc}")
        return 1

    mt5_tf = TIMEFRAME_MAP[timeframe]
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    log(
        "Configuracion: "
        f"START_UTC={start_utc.isoformat()} | END_UTC={end_utc.isoformat()} | "
        f"TIMEFRAME={timeframe} | TICKERS={tickers} | OUTPUT_DIR={output_dir}"
    )

    if not mt5.initialize():
        error_code, error_msg = mt5.last_error()
        log(f"ERROR al inicializar MetaTrader5. Error {error_code}: {error_msg}")
        return 1

    success: list[tuple[str, int, Path]] = []
    no_data: list[str] = []
    failed: list[tuple[str, str]] = []

    try:
        for symbol in tickers:
            try:
                log(f"Descargando {symbol} ({timeframe})...")
                ohlcv = fetch_symbol_ohlcv(symbol, mt5_tf, start_utc, end_utc)

                if ohlcv.empty:
                    no_data.append(symbol)
                    log(f"WARN {symbol}: sin datos en el rango solicitado.")
                    continue

                output_path = output_dir / f"{symbol}.csv"
                write_symbol_csv(ohlcv, output_path)
                success.append((symbol, len(ohlcv), output_path))
                log(f"OK {symbol}: {len(ohlcv)} filas -> {output_path}")
            except Exception as exc:  # noqa: BLE001
                failed.append((symbol, str(exc)))
                log(f"ERROR {symbol}: {exc}")
    finally:
        mt5.shutdown()
        log("Conexion MT5 cerrada.")

    log("Resumen final:")
    log(f"- Exitos: {len(success)}")
    log(f"- Sin datos: {len(no_data)}")
    log(f"- Fallos: {len(failed)}")

    if no_data:
        log(f"  Simbolos sin datos: {', '.join(no_data)}")
    if failed:
        for symbol, reason in failed:
            log(f"  FALLO {symbol}: {reason}")

    if success:
        return 0

    log("No se genero ningun CSV. Exit code = 1.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
