# ================================================================
# data.py — Utilidades + Lectura y sincronización de datos
# (integrado: safe_name, ensure_dirs, erase_directory)
# Base TradingView (CSV) y soporte multi-activo
# ================================================================

import os
import re
import shutil
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------
# LOG muy verboso (puedes mutearlo si no lo necesitas)
# ----------------------------------------------------------------
def _log(msg: str) -> None:
    print(msg)


# =================================================================
# =====================  UTILIDADES COMPARTIDAS  ===================
# =================================================================

def safe_name(file_name: str) -> str:
    """
    Normaliza un nombre para usarlo con seguridad en el sistema de archivos.
    - Elimina la extensión (si existe).
    - Sustituye cualquier carácter que no sea [A-Za-z0-9_-] por "_".

    Logs:
      - Imprime el nombre de entrada y el resultado normalizado.
    """
    base, _ = os.path.splitext(str(file_name))
    out = re.sub(r'[^A-Za-z0-9_\-]+', '_', base)
    _log(f"[DATA][safe_name] in='{file_name}' -> out='{out}'")
    return out
# Explicación:
# Esta función evita errores de escritura/lectura en Windows/Linux por caracteres
# conflictivos (dos puntos, barras, etc.). Quitamos la extensión adrede para que
# tú decidas luego si quieres '.xlsx', '.csv', '.html', etc. según el contexto.


_OUTPUT_DIRS = ["plots", "stats", "trades", "heatmaps", "grid", "cscv", "stress", "wf", "results"]


def project_path(*parts: str) -> str:
    """
    Construye rutas absolutas ancladas al directorio del proyecto.
    """
    return os.path.join(PROJECT_ROOT, *parts)


def output_path(relative_path: str) -> str:
    """
    Devuelve una ruta absoluta dentro del proyecto para un path relativo de salida.
    """
    return project_path(relative_path)

def ensure_dirs() -> None:
    """
    Crea (si no existen) las carpetas de salida típicas del proyecto.
    Carpetas: plots, stats, trades, heatmaps, grid, cscv, stress, wf, results

    Logs:
      - Indica la creación/confirmación de cada carpeta.
    """
    _log("[DATA][ensure_dirs] Verificando carpetas de salida...")
    for folder in _OUTPUT_DIRS:
        folder_path = project_path(folder)
        os.makedirs(folder_path, exist_ok=True)
        _log(f"[DATA][ensure_dirs] OK -> {folder_path}")
    _log("[DATA][ensure_dirs] Completado.")
# Explicación:
# Evita errores de “No such file or directory” al intentar guardar XLSX/HTML/etc.
# Puedes llamarla al inicio del programa (main.py) y antes de cualquier guardado.


def erase_directory() -> None:
    """
    Limpia el CONTENIDO de las carpetas de resultados (no borra la carpeta).
    Afecta a: plots, stats, trades, heatmaps, grid, cscv, stress, wf, results

    Logs:
      - Lista cada item eliminado y resume al final.
    """
    _log("[DATA][erase_directory] Iniciando limpieza de carpetas de resultados...")
    for carpeta in _OUTPUT_DIRS:
        ruta = project_path(carpeta)
        if os.path.isdir(ruta):
            for nombre in os.listdir(ruta):
                p = os.path.join(ruta, nombre)
                try:
                    if os.path.isfile(p) or os.path.islink(p):
                        os.unlink(p)
                        _log(f"[DATA][erase_directory] archivo   BORRADO -> {p}")
                    elif os.path.isdir(p):
                        shutil.rmtree(p)
                        _log(f"[DATA][erase_directory] carpeta   BORRADA -> {p}")
                except Exception as e:
                    _log(f"[DATA][erase_directory][WARN] No se pudo borrar '{p}': {e}")
        else:
            _log(f"[DATA][erase_directory][INFO] No existe -> {ruta}")
    _log("[DATA][erase_directory] LAS CARPETAS HAN SIDO LIMPIADAS.")
# Explicación:
# Para reiniciar experimentos sin restos de ejecuciones anteriores. Sólo elimina
# el contenido interno; las carpetas “madre” se mantienen para no romper rutas.


# =================================================================
# ======================  GESTOR DE DATOS (TV)  ===================
# =================================================================

class DataManager:
    """
    Gestiona:
      - Descubrimiento de carpetas de datos (prefijo 'data_').
      - Enumeración de CSVs por carpeta.
      - Carga de CSV (TradingView), normalización a OHLCV (open/high/low/close/volume)
        y unificación de índice temporal.
      - Validación básica y sincronización multi-activo.

    Notas:
      - Por defecto, interpreta timestamps en UTC y los deja naive (sin tz) tras
        convertir a tu zona si configuras `tz`. Esto evita problemas con librerías
        que esperan índices naive.
    """

    def __init__(self, data_root: str = ".", tz: Optional[str] = None):
        self.data_root = data_root
        self.tz = tz or "UTC"
        _log(f"[DATA] DataManager init: root='{self.data_root}' tz='{self.tz}'")

    # --------------------------------------------------------------
    # Descubrir carpetas data_*
    # --------------------------------------------------------------
    def list_data_folders(self) -> List[str]:
        folders: List[str] = []
        for name in os.listdir(self.data_root):
            p = os.path.join(self.data_root, name)
            if os.path.isdir(p) and (name.startswith("data_") or re.match(r"^data\d+$", name)):
                suffix = name.split("data_", 1)[1]
                if suffix.isdigit():
                    folders.append(p)

        def key_fn(x: str) -> Tuple[int, str]:
            m = re.search(r"([0-9]+)", os.path.basename(x))
            return (int(m.group(1)) if m else 0, x)

        folders.sort(key=key_fn)
        _log(f"[DATA] list_data_folders -> {folders}")
        return folders
    # Explicación:
    # Busca carpetas como data_01, data_02... y las ordena por número para que
    # el procesamiento sea determinista.

    # --------------------------------------------------------------
    # Listar CSVs dentro de una carpeta
    # --------------------------------------------------------------
    def list_csv_files(self, folder: str) -> List[str]:
        folder_path = folder if os.path.isabs(folder) else os.path.join(self.data_root, folder)
        if not os.path.isdir(folder_path):
            _log(f"[DATA][list_csv_files][WARN] Carpeta no encontrada: {folder_path}")
            return []
        files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".csv")
        ]
        files.sort(key=lambda p: os.path.basename(p).lower())
        _log(f"[DATA] list_csv_files('{folder}') -> {len(files)} archivo(s)")
        return files
    # Explicación:
    # Devuelve rutas absolutas de los CSVs de la carpeta. Orden alfabético para
    # reproducibilidad.

    # --------------------------------------------------------------
    # Cargar todos los CSVs de una carpeta (TradingView)
    # --------------------------------------------------------------
    def load_all_from_folder(self, folder: str) -> List[pd.DataFrame]:
        files = self.list_csv_files(folder)
        _log(f"[DATA] load_all_from_folder('{folder}') -> {len(files)} CSV(s)")
        return self.load_tradingview(files)
    # Explicación:
    # Conveniencia: obtiene los CSV y delega la carga a load_tradingview.

    # --------------------------------------------------------------
    # Cargar CSVs TradingView y normalizar a OHLCV
    # --------------------------------------------------------------
    def load_tradingview(self, files: List[str]) -> List[pd.DataFrame]:
        dfs: List[pd.DataFrame] = []

        for path in files:
            _log(f"[DATA][TV] Leyendo -> {path}")
            # 1) Lectura robusta (sep autodetect si es posible)
            try:
                df = pd.read_csv(path, sep=None, engine="python")
            except Exception:
                df = pd.read_csv(path)
            df.columns = [c.strip() for c in df.columns]

            # 2) Detectar columna temporal
            time_col: Optional[str] = None
            for cand in ["time", "Time", "date", "Date", "datetime", "Datetime"]:
                if cand in df.columns:
                    time_col = cand
                    break
            if time_col is None:
                raise ValueError(f"[DATA][TV][ERROR] No se encuentra columna de tiempo en {os.path.basename(path)}")

            ts = df[time_col]

            # 3) Convertir a datetime (UTC) → tz local (self.tz) → naive
            if pd.api.types.is_numeric_dtype(ts):
                # Heurística de unidad: ms si es muy grande, si no segundos
                ma = float(ts.dropna().iloc[0]) if len(ts.dropna()) else 0.0
                unit = "ms" if ma > 1e12 else "s"
                dt = pd.to_datetime(ts, unit=unit, utc=True)
            else:
                dt = pd.to_datetime(ts, utc=True, infer_datetime_format=True, errors="coerce")

            # Si tiene tz, convertimos a la tz pedida y luego dejamos naive
            if getattr(dt.dt, "tz", None) is not None:
                dt = pd.DatetimeIndex(dt).tz_convert(self.tz)
                dt = pd.DatetimeIndex(dt).tz_localize(None)

            df.index = pd.DatetimeIndex(dt)

            # 4) Mapear columnas a estándar OHLCV
            keep_map: Dict[str, List[str]] = {
                "open":   ["open", "Open", "OPEN", "o"],
                "high":   ["high", "High", "HIGH", "h"],
                "low":    ["low", "Low", "LOW", "l"],
                "close":  ["close", "Close", "CLOSE", "c"],
                "volume": ["volume", "Volume", "VOLUME", "vol", "Vol"],
            }

            out = pd.DataFrame(index=df.index.copy())
            for std, candidates in keep_map.items():
                found = None
                for cand in candidates:
                    if cand in df.columns:
                        found = cand
                        break
                if found is not None:
                    out[std] = pd.to_numeric(df[found], errors="coerce")
                else:
                    if std == "volume":
                        out[std] = 0.0
                        _log(f"[DATA][TV][INFO] '{os.path.basename(path)}' sin 'volume' -> volumen=0")
                    else:
                        raise ValueError(f"[DATA][TV][ERROR] Columna '{std}' no encontrada en {os.path.basename(path)}")

            # 4.5) Saneamiento explícito del volumen (relleno de NaN, evitar negativos)
            out = self.fix_volume(out, fill_value=100.0)

            # 5) Ordenar, deduplicar y limpiar NaNs críticos
            before = len(out)
            out = out.sort_index()
            out = out[~out.index.duplicated(keep="last")]
            out = out.dropna(subset=["open", "high", "low", "close"])
            after = len(out)
            if after != before:
                _log(f"[DATA][TV] Filas ajustadas (orden/dup/NaN): {before} -> {after}")

            # 6) Guardar el nombre del archivo en el DataFrame final (atributo personalizado)
            asset_name = os.path.splitext(os.path.basename(path))[0]
            out._asset_name = asset_name

            dfs.append(out)
            _log(f"[DATA][TV] OK -> {os.path.basename(path)} ({len(out)} filas) | _asset_name='{asset_name}'")

        _log(f"[DATA][TV] DataFrames cargados: {len(dfs)}")
        return dfs
    # Explicación:
    # Carga CSV de TradingView (cabecera variable), detecta y convierte tiempos
    # (segundos/milisegundos o strings), normaliza columnas y limpia duplicados/NaNs.
    # Si falta volumen, setea 0. El índice final queda datetime naive.

    # --------------------------------------------------------------
    # Validación de un OHLCV estándar
    # --------------------------------------------------------------
    def validate(self, df: pd.DataFrame) -> bool:
        req = ["open", "high", "low", "close", "volume"]
        ok = (
            isinstance(df.index, pd.DatetimeIndex)
            and not df.empty
            and all(c in df.columns for c in req)
            and df.index.is_monotonic_increasing
        )
        _log(f"[DATA] validate -> {ok}")
        return ok
    # Explicación:
    # Chequeo rápido para evitar pasar objetos inconsistentes a Backtrader.

    def fix_volume(self, df: pd.DataFrame, fill_value: float = 100.0) -> pd.DataFrame:
        """
        Normaliza la columna 'volume' de un OHLCV:
            - Rellena NaN con 'fill_value' (por defecto 100.0).
            - Convierte a numérico (coerce) y recorta negativos a 0 (por seguridad).
            - Devuelve SIEMPRE un DF con 'volume' válido para Backtrader y tus scripts.

        Args:
            df (pd.DataFrame): DataFrame con columnas estándar ['open','high','low','close','volume'].
            fill_value (float): Valor para rellenar NaN de 'volume' (por defecto 100.0).

        Returns:
            pd.DataFrame: Mismo DF (copia) con 'volume' saneado.
        """
        d = df.copy()
        if "volume" not in d.columns:
            # Si no existe, lo creamos a 0.0 para no romper pipelines que esperan la columna
            _log("[DATA][fix_volume][INFO] 'volume' no existe -> creando con 0.0")
            d["volume"] = 0.0
            return d

        # Aseguramos tipo numérico y contabilizamos NaNs previos para log
        d["volume"] = pd.to_numeric(d["volume"], errors="coerce")
        nans_antes = int(d["volume"].isna().sum())

        # Rellenamos NaN y evitamos negativos
        d["volume"] = d["volume"].fillna(float(fill_value))
        d.loc[d["volume"] < 0, "volume"] = 0.0

        if nans_antes > 0:
            _log(f"[DATA][fix_volume] Rellenados {nans_antes} NaN en 'volume' con {fill_value}")

        return d

    # --------------------------------------------------------------
    # Sincronización multi-activo (intersección o union+ffill)
    # --------------------------------------------------------------
    def sync_multi_asset(
        self,
        dfs: List[pd.DataFrame],
        aliases: Optional[List[str]] = None,
        drop_mismatched: bool = True,
        align_frequency: Optional[str] = None
    ) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Sincroniza varias series para operar multi-activo sin desfaces:
          - align_frequency: si se indica, hace resample OHLCV previo (D, H, 15T, 5T, ...).
          - drop_mismatched=True -> intersección estricta de timestamps.
          - drop_mismatched=False -> unión con forward-fill/back-fill (cuidado con look-ahead).

        Devuelve:
          (lista_de_dataframes_sincronizados, aliases_finales)
        """
        def _resample_ohlcv(x: pd.DataFrame, rule: str) -> pd.DataFrame:
            agg = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
            return x.resample(rule, label="right", closed="right").agg(agg).dropna(how="any")

        if not dfs:
            _log("[DATA][sync] Lista de dataframes vacía")
            return [], (aliases or [])

        # 1) Resample si se pide
        out: List[pd.DataFrame] = []
        for i, df in enumerate(dfs):
            d = df.copy()
            if align_frequency:
                _log(f"[DATA][sync] Resample '{aliases[i] if aliases else i}' -> '{align_frequency}'")
                d = _resample_ohlcv(d, align_frequency)
            out.append(d)

        # 2) Intersección o unión de índices
        if drop_mismatched:
            idx = out[0].index
            for d in out[1:]:
                idx = idx.intersection(d.index)
            # Preservar atributos personalizados al reindexar
            new_out = []
            for d in out:
                asset_name = getattr(d, '_asset_name', None)
                d_reindexed = d.reindex(idx)
                if asset_name:
                    d_reindexed._asset_name = asset_name
                new_out.append(d_reindexed)
            out = new_out
            _log(f"[DATA][sync] drop_mismatched=True -> filas comunes = {len(idx)}")
        else:
            full = out[0].index
            for d in out[1:]:
                full = full.union(d.index)
            # Preservar atributos personalizados al reindexar
            new_out = []
            for d in out:
                asset_name = getattr(d, '_asset_name', None)
                d_reindexed = d.reindex(full).ffill().bfill()
                if asset_name:
                    d_reindexed._asset_name = asset_name
                new_out.append(d_reindexed)
            out = new_out
            _log(f"[DATA][sync] drop_mismatched=False -> filas totales = {len(full)} (ffill+bfill)")

        # 3) Aliases
        if aliases is None:
            aliases = [f"asset_{i}" for i in range(len(out))]

        # 4) Validación final
        lens = [len(d) for d in out]
        _log(f"[DATA][sync] Longitudes finales: {lens}")
        return out, aliases
    # Explicación:
    # Para estrategias relativas o carteras: garantiza que todas las series compartan
    # los mismos timestamps. Si trabajas con feeds dispares, usa align_frequency para
    # alinear primero (ej: 'H' o '15T'). drop_mismatched=True evita sesgos al exigir
    # cortes exactos de tiempo; False permite ffill/bfill (con más riesgo).
