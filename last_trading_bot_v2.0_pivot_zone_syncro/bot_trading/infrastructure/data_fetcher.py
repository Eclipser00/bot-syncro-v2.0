"""Servicios de datos de mercado (producción y desarrollo/CSV)."""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from bot_trading.domain.entities import SymbolConfig
from bot_trading.infrastructure.mt5_client import BrokerClient

logger = logging.getLogger(__name__)

_TIMEFRAME_MAP = {
    "M1": "1min",
    "M3": "3min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1H",
    "H4": "4H",
    "D1": "1D",
}

_TF_MINUTES = {
    "M1": 1,
    "M3": 3,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


def _infer_base_minutes(df: pd.DataFrame) -> float | None:
    if df is None or df.empty or len(df.index) < 2:
        return None
    deltas = df.index.to_series().diff().dropna()
    median_sec = deltas.dt.total_seconds().median()
    if median_sec is None or not math.isfinite(median_sec) or median_sec <= 0:
        return None
    return float(median_sec) / 60.0


def _ensure_utc_datetime(value: datetime) -> datetime:
    """Normaliza datetime a UTC-aware para evitar comparaciones naive/aware."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el DatetimeIndex del DataFrame a UTC."""
    if df.empty:
        return df
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        df.index = idx.tz_localize("UTC")
    else:
        df.index = idx.tz_convert("UTC")
    return df


def _resample(df: pd.DataFrame, target_tf: str, symbol_name: str, drop_last_partial: bool = False) -> pd.DataFrame:
    """Resamplea un DataFrame base a un timeframe objetivo.

    Semantica estandar:
    - La vela agregada queda etiquetada en su CIERRE (label='right', closed='right').
    - Si drop_last_partial=True, se descarta la ultima vela solo si esta incompleta.
    """
    freq = _TIMEFRAME_MAP.get(target_tf)
    if freq is None:
        raise ValueError(f"Timeframe solicitado no soportado: {target_tf}")

    resampled = (
        df.resample(freq, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    if drop_last_partial and len(resampled) > 0:
        base_minutes = _infer_base_minutes(df)
        target_minutes = _TF_MINUTES.get(target_tf)
        if base_minutes is None or target_minutes is None or base_minutes <= 0:
            resampled = resampled.iloc[:-1]
        else:
            expected = max(1, int(round(float(target_minutes) / float(base_minutes))))
            last_label = resampled.index[-1]
            bin_start = last_label - pd.Timedelta(minutes=float(target_minutes))
            # Con label/closed=right, el bin es (bin_start, last_label]
            count = df[(df.index > bin_start) & (df.index <= last_label)].shape[0]
            if count < expected:
                resampled = resampled.iloc[:-1]
    resampled.attrs["symbol"] = symbol_name
    return resampled


class BaseDataProvider:
    """Protocolo sencillo para proveedores de datos."""

    def get_data(
        self,
        symbol: SymbolConfig,
        target_timeframes: Iterable[str],
        now: datetime,
    ) -> dict[str, pd.DataFrame]:
        raise NotImplementedError

    def get_simulated_now(self, symbols: Iterable[SymbolConfig]) -> datetime | None:
        """Devuelve un "now" simulado (por ejemplo, desde CSV) si aplica.

        En producción devuelve None.
        """
        return None


class MarketDataService:
    """Fachada simple que delega en un proveedor de datos concreto.

    Para compatibilidad con el código existente, si se le pasa un BrokerClient
    en lugar de un BaseDataProvider, crea internamente un ProductionDataProvider
    con lookbacks mínimos (1 día).
    """

    def __init__(
        self,
        data_provider: "BaseDataProvider | BrokerClient",
        lookback_days_entry: int = 1,
        lookback_days_zone: int = 1,
        lookback_days_stop: int = 1,
    ) -> None:
        if isinstance(data_provider, BaseDataProvider):
            self.data_provider = data_provider
        else:
            # fallback para compatibilidad en tests que pasaban broker directo
            self.data_provider = ProductionDataProvider(
                broker_client=data_provider,
                lookback_days_entry=lookback_days_entry,
                lookback_days_zone=lookback_days_zone,
                lookback_days_stop=lookback_days_stop,
            )

    def get_data(
        self,
        symbol: SymbolConfig,
        target_timeframes: Iterable[str],
        now: datetime,
    ) -> dict[str, pd.DataFrame]:
        return self.data_provider.get_data(symbol, target_timeframes, now)

    # Compatibilidad retro: utilizado por tests existentes
    def get_resampled_data(
        self,
        symbol: SymbolConfig,
        target_timeframes: Iterable[str],
        start: datetime,
        end: datetime,
    ) -> dict[str, pd.DataFrame]:
        if isinstance(self.data_provider, ProductionDataProvider):
            base_tf = symbol.min_timeframe
            raw = self.data_provider.broker_client.get_ohlcv(symbol.name, base_tf, start, end)
            raw.attrs["symbol"] = symbol.name
            result: Dict[str, pd.DataFrame] = {base_tf: raw}
            for tf in set(target_timeframes):
                if tf == base_tf:
                    continue
                result[tf] = _resample(raw, tf, symbol.name, drop_last_partial=False)
            return result

        # Fallback genérico: usa get_data y recorta al rango
        data = self.get_data(symbol, target_timeframes, now=end)
        for tf, df in data.items():
            data[tf] = df.loc[(df.index >= start) & (df.index <= end)]
        return data


class ProductionDataProvider(BaseDataProvider):
    """Obtiene datos desde el broker real y los resamplea."""

    def __init__(
        self,
        broker_client: BrokerClient,
        lookback_days_entry: int,
        lookback_days_zone: int,
        lookback_days_stop: int,
    ) -> None:
        self.broker_client = broker_client
        self.lookback_days_entry = lookback_days_entry
        self.lookback_days_zone = lookback_days_zone
        self.lookback_days_stop = lookback_days_stop
        # Buffers acumulativos por símbolo para que la estrategia detecte nuevas velas.
        self._base_cache: Dict[str, pd.DataFrame] = {}

    def get_data(
        self,
        symbol: SymbolConfig,
        target_timeframes: Iterable[str],
        now: datetime,
    ) -> dict[str, pd.DataFrame]:
        now = _ensure_utc_datetime(now)
        timeframes = set(target_timeframes)
        timeframes.add(symbol.min_timeframe)

        max_days = max(
            [
                self.lookback_days_entry,
                self.lookback_days_zone,
                self.lookback_days_stop,
            ]
        )
        start = now - timedelta(days=max_days)
        end = now

        base_tf = symbol.min_timeframe
        if base_tf not in _TIMEFRAME_MAP:
            raise ValueError(f"Timeframe base no soportado: {base_tf}")

        # Bootstrap inicial: carga ventana completa y guarda en buffer
        if symbol.name not in self._base_cache:
            raw = self.broker_client.get_ohlcv(symbol.name, base_tf, start, end)
            raw = _ensure_utc_index(raw)
            raw.attrs["symbol"] = symbol.name
            self._base_cache[symbol.name] = raw
        else:
            # Incremental: traer solo lo nuevo y concatenar evitando duplicados
            cached = _ensure_utc_index(self._base_cache[symbol.name])
            self._base_cache[symbol.name] = cached
            if cached.empty:
                incremental_start = start
            else:
                incremental_start = _ensure_utc_datetime(cached.index[-1].to_pydatetime())
            raw_inc = self.broker_client.get_ohlcv(symbol.name, base_tf, incremental_start, end)
            raw_inc = _ensure_utc_index(raw_inc)
            if not raw_inc.empty:
                raw_inc.attrs["symbol"] = symbol.name
                concatenated = pd.concat([cached, raw_inc]).sort_index()
                concatenated = concatenated.loc[~concatenated.index.duplicated(keep="last")]
                concatenated.attrs["symbol"] = symbol.name
                self._base_cache[symbol.name] = concatenated

        base_df = self._base_cache[symbol.name]

        result: Dict[str, pd.DataFrame] = {base_tf: base_df}
        for tf in timeframes:
            if tf == base_tf:
                continue
            result[tf] = _resample(base_df, tf, symbol.name, drop_last_partial=True)
        return result


class DevelopmentCsvDataProvider(BaseDataProvider):
    """Simula broker usando CSVs por símbolo y entrega velas cerradas incrementalmente."""

    def __init__(
        self,
        data_dir: Path,
        base_timeframe: str,
        lookback_days_entry: int,
        lookback_days_zone: int,
        lookback_days_stop: int,
    ) -> None:
        self.data_dir = data_dir
        self.base_timeframe = base_timeframe
        self.lookback_days_entry = lookback_days_entry
        self.lookback_days_zone = lookback_days_zone
        self.lookback_days_stop = lookback_days_stop

        self._base_data: Dict[str, pd.DataFrame] = {}
        self._cursor: Dict[str, int] = {}
        self._bootstrapped: Dict[str, bool] = {}
        self._base_tf_by_symbol: Dict[str, str] = {}
        self._last_emitted_cursor: Dict[str, int] = {}

    def get_simulated_now(self, symbols: Iterable[SymbolConfig]) -> datetime | None:
        symbols_list = list(symbols)
        if not symbols_list:
            return None

        ref_symbol = symbols_list[0].name
        self._ensure_loaded(ref_symbol)
        if not self._bootstrapped.get(ref_symbol, False):
            self._bootstrap_cursor(ref_symbol)

        base_df = self._base_data.get(ref_symbol)
        if base_df is None or base_df.empty:
            return None

        cursor = self._cursor.get(ref_symbol, 0)
        if cursor <= 0:
            return base_df.index[0].to_pydatetime()

        cursor = min(cursor, len(base_df))
        return base_df.index[cursor - 1].to_pydatetime()

    def _load_csv(self, symbol: str) -> pd.DataFrame:
        symbol_clean = symbol.split(",")[0].replace("OANDA_", "").strip()
        candidates = [
            self.data_dir / f"{symbol}.csv",
            self.data_dir / f"{symbol_clean}.csv",
            self.data_dir / f"{symbol_clean},.csv",
        ]
        csv_path = next((p for p in candidates if p.exists()), None)
        if csv_path is None:
            wildcard_matches = sorted(self.data_dir.glob(f"{symbol_clean}*.csv"))
            if wildcard_matches:
                csv_path = wildcard_matches[0]
            else:
                raise FileNotFoundError(
                    f"No se encontr? CSV de desarrollo para {symbol} en {self.data_dir / f'{symbol}.csv'}"
                )

        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}
        required = ["open", "high", "low", "close"]
        for req in required:
            if req not in cols:
                raise ValueError(f"CSV {csv_path} no contiene columna {req}")
        volume_col = cols.get("volume")

        if "time" in cols:
            time_series = df[cols["time"]]
            if time_series.iloc[0] > 1e12:
                time_series = pd.to_datetime(time_series, unit="ms", utc=True)
            else:
                time_series = pd.to_datetime(time_series, unit="s", utc=True)
        elif "datetime" in cols:
            time_series = pd.to_datetime(df[cols["datetime"]], utc=True)
        else:
            raise ValueError(f"CSV {csv_path} debe tener columna time o datetime")

        open_arr = pd.to_numeric(df[cols["open"]], errors="coerce").to_numpy()
        high_arr = pd.to_numeric(df[cols["high"]], errors="coerce").to_numpy()
        low_arr = pd.to_numeric(df[cols["low"]], errors="coerce").to_numpy()
        close_arr = pd.to_numeric(df[cols["close"]], errors="coerce").to_numpy()
        if volume_col:
            volume_arr = pd.to_numeric(df[volume_col], errors="coerce").to_numpy()
        else:
            volume_arr = pd.Series([0.0] * len(df)).to_numpy()

        ohlcv = pd.DataFrame(
            {
                "open": open_arr,
                "high": high_arr,
                "low": low_arr,
                "close": close_arr,
                "volume": volume_arr,
            },
            index=pd.DatetimeIndex(time_series),
        )
        ohlcv = ohlcv.sort_index()
        ohlcv = ohlcv[~ohlcv.index.duplicated(keep="last")]
        ohlcv.attrs["symbol"] = symbol
        return ohlcv

    def _infer_timeframe(self, df: pd.DataFrame) -> str:
        """Detecta timeframe base a partir del delta temporal mediano."""
        if len(df.index) < 2:
            return self.base_timeframe
        deltas = df.index.to_series().diff().dropna()
        median_sec = deltas.dt.total_seconds().median()
        if median_sec is None or not math.isfinite(median_sec):
            return self.base_timeframe
        minutes = median_sec / 60.0
        # Elegir el tf más cercano de la tabla conocida
        best_tf = self.base_timeframe
        best_diff = float("inf")
        for tf, mins in _TF_MINUTES.items():
            diff = abs(minutes - mins)
            if diff < best_diff:
                best_diff = diff
                best_tf = tf
        return best_tf

    def _ensure_loaded(self, symbol: str) -> None:
        if symbol in self._base_data:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._base_data[symbol] = self._load_csv(symbol)
        self._cursor[symbol] = 0
        self._bootstrapped[symbol] = False
        self._base_tf_by_symbol[symbol] = self._infer_timeframe(self._base_data[symbol])

    def _bootstrap_cursor(self, symbol: str) -> None:
        base_df = self._base_data[symbol]
        if base_df.empty:
            self._cursor[symbol] = 0
            self._bootstrapped[symbol] = True
            return

        base_tf = self._base_tf_by_symbol.get(symbol, self.base_timeframe)
        minutes = _TF_MINUTES.get(base_tf, 1)
        max_days = max(self.lookback_days_entry, self.lookback_days_zone, self.lookback_days_stop)
        bars_needed = max(1, int(math.ceil((max_days * 24 * 60) / minutes)))
        if bars_needed >= len(base_df):
            self._cursor[symbol] = 1
        else:
            self._cursor[symbol] = bars_needed
        self._bootstrapped[symbol] = True

    def _current_slice(self, symbol: str) -> pd.DataFrame:
        base_df = self._base_data[symbol]
        cursor = self._cursor.get(symbol, 0)
        return base_df.iloc[:cursor].copy()

    def _advance(self, symbol: str, steps: int = 1) -> None:
        base_df = self._base_data[symbol]
        new_cursor = min(len(base_df), self._cursor.get(symbol, 0) + steps)
        self._cursor[symbol] = new_cursor

    def get_data(
        self,
        symbol: SymbolConfig,
        target_timeframes: Iterable[str],
        now: datetime,
    ) -> dict[str, pd.DataFrame]:
        self._ensure_loaded(symbol.name)
        if not self._bootstrapped.get(symbol.name, False):
            self._bootstrap_cursor(symbol.name)

        base_df = self._base_data[symbol.name]
        if base_df.empty:
            raise StopIteration(f"CSV sin datos para {symbol.name}")

        cursor = self._cursor.get(symbol.name, 0)
        last_emitted = self._last_emitted_cursor.get(symbol.name, -1)
        if cursor >= len(base_df) and last_emitted == cursor:
            raise StopIteration(f"CSV agotado para {symbol.name}")
        self._last_emitted_cursor[symbol.name] = cursor

        base_slice = base_df.iloc[:cursor].copy()
        base_slice.attrs["symbol"] = symbol.name

        result: Dict[str, pd.DataFrame] = {}
        for tf in set(target_timeframes).union({self.base_timeframe}):
            if tf == self.base_timeframe or tf == self._base_tf_by_symbol.get(symbol.name, self.base_timeframe):
                result[tf] = base_slice
            else:
                # En streaming desde CSV, el resample de TF superiores genera una "última vela" que
                # todavía se está formando (parcial) y cuyos OHLC pueden cambiar con la siguiente vela base.
                # Para evitar divergencias vs Backtrader/producción, descartamos esa última vela parcial.
                result[tf] = _resample(base_slice, tf, symbol.name, drop_last_partial=True)

        # Avanzar 1 vela base para la próxima llamada (streaming vela a vela)
        self._advance(symbol.name, steps=1)
        return result
