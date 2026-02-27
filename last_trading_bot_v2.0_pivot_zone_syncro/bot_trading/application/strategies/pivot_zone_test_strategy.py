"""Port de la estrategia PivotZoneTest (Backtrader) a last_trading_bot_v1.0.

Implementa:
- Construcción y persistencia de zonas pivote (TF_zone) con ATR TA-Lib y pivotes 3 velas.
- Entradas por ruptura sobre TF_entry contra cualquier zona guardada con filtros.
- Target zone-to-zone (TP) usando la zona existente más cercana en dirección.
- Stop inicial con pivotes confirmados en TF_stop.
- Gestión de salidas mediante órdenes pendientes (bracket TP/SL) en el broker; sin cierres por toque intrabar.

Notas de integración con el bot:
- El motor ejecuta estrategias por símbolo y timeframe. Esta estrategia requiere varios timeframes.
- Para calcular lot_size (Signal.size) de forma robusta y para poder cerrar con volumen válido,
  es recomendable inyectar broker_client (MetaTrader5Client o FakeBroker con get_open_positions/get_account_info).
"""
from __future__ import annotations

import logging
from pathlib import Path
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import math
import statistics
import pandas as pd
try:
    import talib  # type: ignore
except Exception:
    talib = None

from bot_trading.application.engine.signals import Signal, SignalType
from typing import TYPE_CHECKING
from bot_trading.application.utils.event_logger import EventLogger

if TYPE_CHECKING:
    from bot_trading.infrastructure.mt5_client import BrokerClient
else:
    BrokerClient = Any  # type: ignore[misc,assignment]


logger = logging.getLogger(__name__)
pivot_logger = logging.getLogger("pivot_zones")

_TF_ORDER = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
_EVENT_LOGGER = EventLogger(str(Path(__file__).resolve().parents[4] / "outputs" / "bot_events.jsonl"))


def _magic_number_from_strategy_name(strategy_name: str) -> int:
    """Replica StrategyRegistry.register_strategy(): md5[:8] -> int32 positivo."""
    import hashlib

    hash_hex = hashlib.md5(strategy_name.encode("utf-8")).hexdigest()[:8]
    return int(hash_hex, 16) % (2**31)


def _drop_incomplete_last_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo velas cerradas.

    En este bot, el scheduler intenta correr tras cierre, pero por seguridad
    descartamos la última fila, que podría ser una vela aún en formación.
    """
    if df is None or df.empty:
        return df
    return df.iloc[:-1].copy()


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _format_ts(ts: Any) -> str:
    """Convierte timestamps de índice a string ISO seguro para logs."""
    if ts is None:
        return "NA"
    try:
        if hasattr(ts, "isoformat"):
            return ts.isoformat()
    except Exception:
        pass
    return str(ts)


@dataclass
class _Pivot3CandleState:
    """Detector incremental de pivotes confirmados con 3 velas (sin look-ahead).

    IMPORTANTE (paridad con Backtrader):
    Backtrader (ver `Pivot3Candle` del backtest) confirma el pivote con un retardo de 1 vela
    adicional: en la vela cerrada i, evalua el patron usando las 3 velas PREVIAS (i-3, i-2, i-1),
    y por tanto confirma el pivote en la vela central (i-2):

      - pivot_max si: high[i-3] < high[i-2] > high[i-1]  -> valor high[i-2]
      - pivot_min si:  low[i-3]  >  low[i-2]  < low[i-1] -> valor low[i-2]

    Devuelve (pivot_max, pivot_min) en el instante de confirmación, si existe.
    """
    highs: List[float] = field(default_factory=list)
    lows: List[float] = field(default_factory=list)

    def update(self, high: float, low: float) -> Tuple[Optional[float], Optional[float]]:
        self.highs.append(float(high))
        self.lows.append(float(low))
        # Necesitamos al menos 4 velas acumuladas para poder evaluar (i-3, i-2, i-1) cuando
        # acabamos de incorporar la vela i (la actual del loop). Esto evita adelantar pivotes
        # respecto al backtest (que NO usa la vela actual para confirmarlos).
        if len(self.highs) < 4:
            return (None, None)

        # Paridad exacta con `Pivot3Candle.next()` del backtest:
        # usa las 3 velas previas a la actual (excluye la vela recien anadida).
        h3, h2, h1 = self.highs[-4], self.highs[-3], self.highs[-2]
        l3, l2, l1 = self.lows[-4], self.lows[-3], self.lows[-2]

        pivot_max = h2 if (h2 > h3) and (h2 > h1) else None
        pivot_min = l2 if (l2 < l3) and (l2 < l1) else None
        return (pivot_max, pivot_min)


@dataclass
class _PivotZoneProject:
    """Estado incremental tipo indicador PivotZone (sin look-ahead)."""
    pivots: List[float] = field(default_factory=list)
    mid: float = float("nan")

    has_been_above: bool = False
    has_been_below: bool = False

    locked: bool = False
    locked_mid: float = float("nan")
    locked_width: float = float("nan")

    def reset(self) -> None:
        self.pivots = []
        self.mid = float("nan")
        self.has_been_above = False
        self.has_been_below = False
        self.locked = False
        self.locked_mid = float("nan")
        self.locked_width = float("nan")

    def add_pivot(self, price: float) -> None:
        self.pivots.append(float(price))
        self.mid = float(statistics.median(self.pivots)) if self.pivots else float("nan")


@dataclass
class _TradeState:
    """Estado persistente de un trade para gestión de salidas internas."""
    in_position: bool = False
    direction: int = 0  # +1 long, -1 short
    entry_timeframe: Optional[str] = None

    broken_zone: Optional[Dict[str, float]] = None
    target_zone: Optional[Dict[str, float]] = None

    active_stop: Optional[float] = None
    active_tp: Optional[float] = None

    tp_order_id: Optional[int] = None
    sl_order_id: Optional[int] = None
    stop_update_pending: bool = False
    pending_stop_price: Optional[float] = None


@dataclass
class _PivotParams:
    """Parámetros efectivos (por símbolo) para PivotZoneTest."""
    n1: int
    n2: int
    n3: int
    size_pct: float
    p: float


@dataclass
class PivotZoneTestStrategy:
    """Port de PivotZoneTest (Backtrader) al contrato del bot.

    Timeframes:
      - tf_entry: timeframe principal de decisión y gestión interna de SL/TP.
      - tf_zone: timeframe superior para construir zonas pivote.
      - tf_stop: timeframe para pivotes que definen el stop inicial.

    """

    # Requeridos por el motor
    name: str
    timeframes: List[str] = field(default_factory=list)

    # Integración opcional
    broker_client: Optional[BrokerClient] = None
    allowed_symbols: Optional[List[str]] = None
    symbol_params: Dict[str, Dict[str, float]] = field(default_factory=dict)  # overrides por símbolo

    # Config multi-TF
    tf_entry: str = "M1"
    tf_zone: str = "M3"
    tf_stop: str = "M1"

    # Parámetros Backtrader (mantener nombres)
    n1: int = 3
    n2: int = 60
    n3: int = 3
    size_pct: float = 0.05
    p: float = 0.50

    # Estado persistente por símbolo (no se borra)
    _saved_zones_by_symbol: Dict[str, List[Dict[str, float]]] = field(default_factory=dict, init=False)
    _zone_project_by_symbol: Dict[str, _PivotZoneProject] = field(default_factory=dict, init=False)

    _pivot_zone_by_symbol: Dict[str, _Pivot3CandleState] = field(default_factory=dict, init=False)
    _pivot_stop_by_symbol: Dict[str, _Pivot3CandleState] = field(default_factory=dict, init=False)

    _last_len_zone: Dict[str, int] = field(default_factory=dict, init=False)
    _last_len_stop: Dict[str, int] = field(default_factory=dict, init=False)
    _prev_valid_by_symbol: Dict[str, float] = field(default_factory=dict, init=False)

    _stop_last_min: Dict[str, Optional[float]] = field(default_factory=dict, init=False)
    _stop_last_max: Dict[str, Optional[float]] = field(default_factory=dict, init=False)
    _stop_pivot_mins: Dict[str, List[float]] = field(default_factory=dict, init=False)
    _stop_pivot_maxs: Dict[str, List[float]] = field(default_factory=dict, init=False)

    _trade_state: Dict[str, _TradeState] = field(default_factory=dict, init=False)
    _atr_period: int = field(default=14, init=False)

    def __post_init__(self) -> None:
        # Publicar timeframes requeridos al motor.
        # Si el usuario ya pasó `timeframes`, no los pisamos: solo añadimos los que falten.
        required = {self.tf_entry, self.tf_zone, self.tf_stop}

        if not self.timeframes:
            self.timeframes = sorted(required, key=lambda x: _TF_ORDER.index(x) if x in _TF_ORDER else 999)
            return

        missing = [tf for tf in required if tf not in set(self.timeframes)]
        if missing:
            self.timeframes.extend(sorted(missing, key=lambda x: _TF_ORDER.index(x) if x in _TF_ORDER else 999))

    def _emit_event(self, payload: Dict[str, Any]) -> None:
        """Wrapper seguro para escribir eventos JSONL sin romper la ejecución."""
        try:
            _EVENT_LOGGER.log(payload)
        except Exception:
            pass

    def _resolve_params(self, symbol: str) -> _PivotParams:
        """Combina parámetros base con overrides específicos del símbolo."""
        overrides = self.symbol_params.get(symbol, {}) if self.symbol_params else {}
        params = _PivotParams(
            n1=int(overrides.get("n1", self.n1)),
            n2=int(overrides.get("n2", self.n2)),
            n3=int(overrides.get("n3", self.n3)),
            size_pct=float(overrides.get("size_pct", self.size_pct)),
            p=float(overrides.get("p", self.p)),
        )
        if overrides:
            logger.debug(
                "[%s] Overrides por símbolo aplicados a %s: %s",
                self.name,
                symbol,
                overrides,
            )
        return params

    # -----------------------------
    # Helpers de broker / símbolo
    # -----------------------------
    def _get_symbol(self, data_by_timeframe: Dict[str, pd.DataFrame]) -> Optional[str]:
        # En este repo MarketDataService rellena df.attrs["symbol"] siempre.
        for tf in self.timeframes:
            df = data_by_timeframe.get(tf)
            if df is not None:
                sym = df.attrs.get("symbol")
                if sym:
                    return str(sym)
        return None

    def _get_magic_number(self) -> int:
        return _magic_number_from_strategy_name(self.name)

    def _get_open_position(self, symbol: str):
        """Devuelve la Position abierta para (symbol, magic_number) si existe."""
        if self.broker_client is None:
            return None
        try:
            positions = self.broker_client.get_open_positions()
        except Exception as e:
            logger.error("[%s] Error consultando posiciones abiertas: %s", self.name, e)
            return None

        magic = self._get_magic_number()
        for p in positions:
            if getattr(p, "symbol", None) == symbol and getattr(p, "magic_number", None) == magic:
                return p
        return None

    def _get_equity(self) -> Optional[float]:
        if self.broker_client is None:
            return None
        try:
            info = self.broker_client.get_account_info()
            return _safe_float(getattr(info, "equity", None))
        except Exception:
            return None

    def _get_symbol_info(self, symbol: str):
        """Obtiene info de símbolo de MT5 (si el broker_client lo soporta)."""
        if self.broker_client is None:
            return None
        # MetaTrader5Client lo expone como método "privado" pero disponible.
        if hasattr(self.broker_client, "_get_symbol_info"):
            try:
                return self.broker_client._get_symbol_info(symbol)  # type: ignore[attr-defined]
            except Exception as e:
                logger.debug("[%s] No se pudo obtener symbol_info para %s: %s", self.name, symbol, e)
                return None
        return None

    # -----------------------------
    # Helpers de TF y datos
    # -----------------------------
    def _validate_timeframes(
        self, data_by_timeframe: Dict[str, pd.DataFrame]
    ) -> Tuple[bool, Optional[str]]:
        missing = [tf for tf in self.timeframes if tf not in data_by_timeframe or data_by_timeframe[tf] is None]
        if missing:
            return (False, f"Faltan timeframes requeridos: {missing}")
        return (True, None)

    def _closed_series(self, df: pd.DataFrame) -> pd.DataFrame:
        # Los proveedores de datos de desarrollo y producciÃ³n entregan velas cerradas;
        # evitamos descartar la Ãºltima fila para no perder eventos (alineado con backtest).
        return df

    def _close_pos_ratio(self, row: pd.Series) -> float:
        """pos = (close-low)/(high-low), con fallback 0.5 si rango 0."""
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        rng = h - l
        if rng <= 0.0:
            return 0.5
        return (c - l) / rng

    def _atr_series(self, df: pd.DataFrame, n1: int) -> Optional[Any]:
        """Devuelve array ATR alineado con df (mismo largo) o None si no hay talib."""
        if talib is None:
            return None
        high = df["high"].astype(float).to_numpy()
        low = df["low"].astype(float).to_numpy()
        close = df["close"].astype(float).to_numpy()
        return talib.ATR(high, low, close, timeperiod=int(n1))

    # -----------------------------
    # Construcción incremental zonas (TF_zone)
    # -----------------------------
    def _ensure_symbol_state(self, symbol: str) -> None:
        if symbol not in self._saved_zones_by_symbol:
            self._saved_zones_by_symbol[symbol] = []
        if symbol not in self._zone_project_by_symbol:
            self._zone_project_by_symbol[symbol] = _PivotZoneProject()
        if symbol not in self._pivot_zone_by_symbol:
            self._pivot_zone_by_symbol[symbol] = _Pivot3CandleState()
        if symbol not in self._pivot_stop_by_symbol:
            self._pivot_stop_by_symbol[symbol] = _Pivot3CandleState()
        if symbol not in self._trade_state:
            self._trade_state[symbol] = _TradeState()
        if symbol not in self._prev_valid_by_symbol:
            self._prev_valid_by_symbol[symbol] = 0.0
        if symbol not in self._stop_pivot_mins:
            self._stop_pivot_mins[symbol] = []
        if symbol not in self._stop_pivot_maxs:
            self._stop_pivot_maxs[symbol] = []

    def _process_zone_tf(self, symbol: str, df_zone_closed: pd.DataFrame, params: _PivotParams) -> None:
        """Actualiza PivotZone incremental y guarda zonas en transición valid 0->1."""
        self._ensure_symbol_state(symbol)

        last_len = self._last_len_zone.get(symbol, 0)
        if last_len >= len(df_zone_closed):
            logger.debug(
                "[%s] Sin barras nuevas para zonas pivote (%s). total=%d",
                self.name,
                symbol,
                len(df_zone_closed),
            )
            return

        piv_state = self._pivot_zone_by_symbol[symbol]
        proj = self._zone_project_by_symbol[symbol]
        saved = self._saved_zones_by_symbol[symbol]
        prev_valid = float(self._prev_valid_by_symbol.get(symbol, 0.0))

        atr_arr = self._atr_series(df_zone_closed, self._atr_period)
        if atr_arr is None:
            logger.debug("[%s] ATR no disponible, usando ancho fallback (%s)", self.name, symbol)

        logger.debug(
            "[%s] Procesando zonas pivote para %s: nuevas=%d total=%d",
            self.name,
            symbol,
            len(df_zone_closed) - last_len,
            len(df_zone_closed),
        )

        # Paridad con Backtrader:
        # El indicador PivotZone del backtest usa `addminperiod(max(3, n1))`, lo cual impide
        # que el proyecto de zona (pivotes + flags de role) se construya antes de disponer
        # de suficiente historial para ATR. Mantenemos el detector de pivotes actualizado
        # desde el inicio, pero NO alimentamos el proyecto de zona hasta alcanzar minperiod.
        min_period = max(3, int(self._atr_period))

        for i in range(last_len, len(df_zone_closed)):
            row = df_zone_closed.iloc[i]
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            bar_time = _format_ts(getattr(row, "name", None))

            # Mantener pivotes "calientes" siempre (igual que Pivot3Candle corre desde minperiod=3),
            # pero solo usar sus señales para construir la zona cuando PivotZone ya puede correr.
            pivot_max, pivot_min = piv_state.update(high=high, low=low)
            if (i + 1) < min_period:
                continue

            width = None
            if atr_arr is not None:
                atr_val = float(atr_arr[i])
                if math.isfinite(atr_val) and atr_val > 0:
                    width = atr_val * (float(params.n2) / 100.0)
            if width is None or not math.isfinite(width) or width <= 0:
                width = abs(close) * 0.001  # 0.1%

            pivot_price = None
            pivot_reason = ""
            if pivot_max is not None:
                pivot_price = float(pivot_max)
                pivot_reason = "pivot_max"
            elif pivot_min is not None:
                pivot_price = float(pivot_min)
                pivot_reason = "pivot_min"

            if pivot_price is not None and not proj.locked:
                try:
                    self._emit_event(
                        {
                            "ts_event": bar_time,
                            "bar_index": int(i),
                            "symbol": symbol,
                            "timeframe": self.tf_zone,
                            "event_type": "pivot_confirmed",
                            "side": "flat",
                            "price_ref": "pivot",
                            "price": float(pivot_price),
                            "size": 0.0,
                            "commission": 0.0,
                            "slippage": 0.0,
                            "reason": pivot_reason or "pivot",
                            "order_id": "",
                        }
                    )
                except Exception:
                    pass
                logger.debug(
                    "[%s] Pivote confirmado para %s: precio=%.5f bar=%d ancho=%.5f",
                    self.name,
                    symbol,
                    pivot_price,
                    i,
                    width,
                )
                if not proj.pivots:
                    proj.add_pivot(pivot_price)
                    logger.debug(
                        "[%s] Proyecto zona iniciado para %s: pivots=%d mid=%.5f",
                        self.name,
                        symbol,
                        len(proj.pivots),
                        float(proj.mid),
                    )
                else:
                    if math.isfinite(proj.mid) and abs(pivot_price - float(proj.mid)) <= width:
                        proj.add_pivot(pivot_price)
                        logger.debug(
                            "[%s] Pivote agregado dentro de ancho para %s: pivots=%d mid=%.5f",
                            self.name,
                            symbol,
                            len(proj.pivots),
                            float(proj.mid),
                        )
                    else:
                        prev_mid = float(proj.mid)
                        proj.pivots = [float(pivot_price)]
                        proj.mid = float(statistics.median(proj.pivots))
                        logger.debug(
                            "[%s] Pivote fuera de ancho, reinicio pivots %s: mid_prev=%.5f pivot=%.5f ancho=%.5f",
                            self.name,
                            symbol,
                            prev_mid,
                            pivot_price,
                            width,
                        )

            if math.isfinite(float(proj.mid)):
                top = float(proj.mid) + float(width)
                bot = float(proj.mid) - float(width)
                if close > top:
                    proj.has_been_above = True
                elif close < bot:
                    proj.has_been_below = True

            if (not proj.locked) and (len(proj.pivots) >= int(params.n3)) and proj.has_been_above and proj.has_been_below:
                proj.locked = True
                proj.locked_mid = float(proj.mid)
                proj.locked_width = float(width)
                logger.info(
                    "[%s] Zona bloqueada para %s: mid=%.5f ancho=%.5f pivots=%d role_above=%s role_below=%s",
                    self.name,
                    symbol,
                    float(proj.mid),
                    float(width),
                    len(proj.pivots),
                    proj.has_been_above,
                    proj.has_been_below,
                )

            if os.getenv("PARITY_ZONE_STATE", "0") == "1":
                try:
                    max_bars = int(os.getenv("PARITY_ZONE_STATE_MAX", "0") or 0)
                except Exception:
                    max_bars = 0
                if max_bars <= 0 or (i + 1) <= max_bars:
                    mid_val = proj.locked_mid if proj.locked else proj.mid
                    width_val = proj.locked_width if proj.locked else width
                    reason = (
                        f"valid={int(proj.locked)} "
                        f"above={int(proj.has_been_above)} "
                        f"below={int(proj.has_been_below)} "
                        f"pivots={len(proj.pivots)}"
                    )
                    try:
                        self._emit_event(
                            {
                                "ts_event": bar_time,
                                "bar_index": int(i),
                                "symbol": symbol,
                                "timeframe": self.tf_zone,
                            "event_type": "zone_state",
                            "side": "flat",
                            "price_ref": "mid",
                            "price": float(mid_val) if math.isfinite(float(mid_val)) else None,
                            "size": float(width_val) if math.isfinite(float(width_val)) else None,
                            "state_valid": int(proj.locked),
                            "state_has_been_above": int(proj.has_been_above),
                            "state_has_been_below": int(proj.has_been_below),
                            "state_pivots": int(len(proj.pivots)),
                            "state_mid": float(mid_val) if math.isfinite(float(mid_val)) else None,
                            "state_width": float(width_val) if math.isfinite(float(width_val)) else None,
                            "commission": 0.0,
                            "slippage": 0.0,
                            "reason": reason,
                            "order_id": "",
                        }
                        )
                    except Exception:
                        pass

            current_valid = 1.0 if proj.locked else 0.0
            if current_valid == 1.0 and prev_valid == 0.0:
                if math.isfinite(proj.locked_mid) and math.isfinite(proj.locked_width):
                    zone_top = float(proj.locked_mid) + float(proj.locked_width)
                    zone_bot = float(proj.locked_mid) - float(proj.locked_width)
                    new_zone = {"top": zone_top, "bot": zone_bot, "bar": float(i + 1)}

                    new_mid = (new_zone["top"] + new_zone["bot"]) / 2.0
                    new_width = (new_zone["top"] - new_zone["bot"])
                    min_distance = new_width * max(0.0, float(params.n1))

                    too_close = False
                    for z in saved:
                        z_mid = (float(z["top"]) + float(z["bot"])) / 2.0
                        if abs(new_mid - z_mid) < min_distance:
                            too_close = True
                            break

                    if not too_close:
                        saved.append(new_zone)
                        logger.info(
                            "[%s] ZONA_GUARDADA %s | bar=%d top=%.8f bot=%.8f total=%d",
                            self.name,
                            symbol,
                            int(new_zone["bar"]),
                            new_zone["top"],
                            new_zone["bot"],
                            len(saved),
                        )
                        try:
                            self._emit_event(
                                {
                                    "ts_event": bar_time,
                                    "bar_index": int(i),
                                    "symbol": symbol,
                                    "timeframe": self.tf_zone,
                                    "event_type": "zone_saved",
                                    "side": "flat",
                                    "price_ref": "mid",
                                    "price": float(new_mid),
                                    "size": float(new_width),
                                    "commission": 0.0,
                                    "slippage": 0.0,
                                    "reason": "zone_saved",
                                    "order_id": "",
                                }
                            )
                        except Exception:
                            pass
                        pivot_logger.info(
                            "time=%s symbol=%s tf=%s top=%.8f bot=%.8f mid=%.8f width=%.8f pivots=%d bar=%d total_saved=%d",
                            bar_time,
                            symbol,
                            self.tf_zone,
                            new_zone["top"],
                            new_zone["bot"],
                            float(proj.locked_mid),
                            float(proj.locked_width),
                            len(proj.pivots),
                            int(new_zone["bar"]),
                            len(saved),
                        )
                    else:
                        logger.debug(
                            "[%s] Zona rechazada por cercania para %s: top=%.5f bot=%.5f min_dist=%.5f",
                            self.name,
                            symbol,
                            new_zone["top"],
                            new_zone["bot"],
                            min_distance,
                        )

                proj.reset()
                prev_valid = 0.0
            else:
                prev_valid = current_valid

        self._prev_valid_by_symbol[symbol] = prev_valid
        self._last_len_zone[symbol] = len(df_zone_closed)

    # -----------------------------
    # Pivotes de stop inicial (TF_stop)
    # -----------------------------
    def _process_stop_tf(self, symbol: str, df_stop_closed: pd.DataFrame) -> None:
        self._ensure_symbol_state(symbol)

        last_len = self._last_len_stop.get(symbol, 0)
        if last_len >= len(df_stop_closed):
            return

        piv_state = self._pivot_stop_by_symbol[symbol]
        last_min = self._stop_last_min.get(symbol)
        last_max = self._stop_last_max.get(symbol)

        for i in range(last_len, len(df_stop_closed)):
            row = df_stop_closed.iloc[i]
            high = float(row["high"])
            low = float(row["low"])

            pivot_max, pivot_min = piv_state.update(high=high, low=low)
            if pivot_min is not None:
                last_min = float(pivot_min)
                self._stop_pivot_mins[symbol].append(last_min)
            if pivot_max is not None:
                last_max = float(pivot_max)
                self._stop_pivot_maxs[symbol].append(last_max)

        self._stop_last_min[symbol] = last_min
        self._stop_last_max[symbol] = last_max
        self._last_len_stop[symbol] = len(df_stop_closed)

    def _find_stop_inside_broken_zone(
        self, symbol: str, broken_zone: Dict[str, float], direction: int
    ) -> Optional[float]:
        """Busca el último pivote confirmado de TF_stop que cae dentro de la zona rota."""
        top = float(broken_zone["top"])
        bot = float(broken_zone["bot"])
        pivots = self._stop_pivot_mins.get(symbol, []) if direction > 0 else self._stop_pivot_maxs.get(symbol, [])
        for price in reversed(pivots):
            p = float(price)
            if bot <= p <= top:
                return p
        return None

    # -----------------------------
    # Entrada: ruptura de cualquier zona guardada
    # -----------------------------
    def _select_target_zone(
        self, zones: List[Dict[str, float]], broken_zone: Dict[str, float], direction: int
    ) -> Optional[Dict[str, float]]:
        broken_mid = (float(broken_zone["top"]) + float(broken_zone["bot"])) / 2.0
        best = None
        best_dist = None

        for z in zones:
            if z is broken_zone:
                continue
            mid = (float(z["top"]) + float(z["bot"])) / 2.0
            if direction > 0 and mid > broken_mid:
                dist = abs(mid - broken_mid)
            elif direction < 0 and mid < broken_mid:
                dist = abs(mid - broken_mid)
            else:
                continue

            if best is None or (best_dist is not None and dist < best_dist):
                best = z
                best_dist = dist

        return best

    def _tp_edge_nearest(self, target_zone: Dict[str, float], entry_price: float) -> float:
        top = float(target_zone["top"])
        bot = float(target_zone["bot"])
        d_top = abs(top - float(entry_price))
        d_bot = abs(bot - float(entry_price))
        return bot if d_bot <= d_top else top

    def _check_breakout(
        self, df_entry_closed: pd.DataFrame, zones: List[Dict[str, float]], p: float
    ) -> Optional[Tuple[Dict[str, float], int]]:
        if not zones:
            logger.debug("[%s] Sin zonas guardadas: no se evalúa breakout", self.name)
            return None
        if len(df_entry_closed) < 5:
            logger.debug(
                "[%s] Datos insuficientes en TF_entry (%d velas cerradas): no se evalúa breakout",
                self.name,
                len(df_entry_closed),
            )
            return None

        # i0 = vela actual cerrada, i1.. = previas cerradas
        r0 = df_entry_closed.iloc[-1]
        r1 = df_entry_closed.iloc[-2]
        r2 = df_entry_closed.iloc[-3]
        r3 = df_entry_closed.iloc[-4]

        c0 = float(r0["close"])
        c1 = float(r1["close"])

        for zone in zones:
            top = float(zone["top"])
            bot = float(zone["bot"])

            # Validar origen: >=2 de las 3 velas cerradas anteriores dentro de [bot, top]
            inside_count = 0
            for rr in (r1, r2, r3):
                cc = float(rr["close"])
                if bot <= cc <= top:
                    inside_count += 1
            if inside_count < 2:
                logger.debug(
                    "[%s] Breakout descartado por origen: %s inside_count=%d (<2 de 3)",
                    self.name,
                    symbol := df_entry_closed.attrs.get("symbol", "UNKNOWN"),
                    inside_count,
                )
                continue

            # LONG: confirmación de ruptura por 2 cierres consecutivos fuera de zona.
            long_ok = (float(r1["close"]) > top) and (c0 > top)
            if long_ok:
                target = self._select_target_zone(zones, zone, +1)
                if target is not None:
                    return (zone, +1)
                logger.debug(
                    "[%s] Breakout LONG descartado: sin target available (zones=%d)",
                    self.name,
                    len(zones),
                )

            # SHORT: confirmación de ruptura por 2 cierres consecutivos fuera de zona.
            short_ok = (float(r1["close"]) < bot) and (c0 < bot)
            if short_ok:
                target = self._select_target_zone(zones, zone, -1)
                if target is not None:
                    return (zone, -1)
                logger.debug(
                    "[%s] Breakout SHORT descartado: sin target available (zones=%d)",
                    self.name,
                    len(zones),
                )

        logger.debug(
            "[%s] Sin breakout: close=%.5f prev_close=%.5f zonas=%d",
            self.name,
            float(c0),
            float(c1),
            len(zones),
        )
        return None

    # -----------------------------
    # Sizing: riesgo por stop (size_pct)
    # -----------------------------
    def _calc_lot_size_by_stop(
        self, symbol: str, entry_price: float, stop_price: float, size_pct: float
    ) -> Optional[float]:
        """Calcula tamaño en lotes con `size_pct` como nocional objetivo.

        Política:
        - Base: nocional objetivo ~= equity * size_pct.
        - Modulación por stop: el lote se ajusta con un factor suave en función de
          la distancia relativa al stop (más estrecho -> más tamaño; más ancho -> menos).
        - Cap nocional superior: evita exposiciones desproporcionadas.
        """
        equity = self._get_equity()
        if equity is None or equity <= 0:
            logger.debug("[%s] Size descartado: equity no disponible o <=0 (equity=%s)", self.name, equity)
            return None

        dist = abs(float(entry_price) - float(stop_price))
        if dist <= 0:
            logger.debug("[%s] Size descartado: distancia stop=0 (entry=%.5f stop=%.5f)", self.name, entry_price, stop_price)
            return None

        if os.getenv("PARITY_SIZING", "0") == "1":
            # Paridad: misma semántica que el backtest (size en unidades nocionales).
            try:
                leverage = float(os.getenv("PARITY_LEVERAGE", "1"))
            except Exception:
                leverage = 1.0
            leverage = max(1.0, leverage)
            try:
                stop_ref_pct = float(os.getenv("SIZE_STOP_REFERENCE_PCT", "0.00075"))
            except Exception:
                stop_ref_pct = 0.00075
            try:
                factor_min = float(os.getenv("SIZE_STOP_FACTOR_MIN", "0.80"))
                factor_max = float(os.getenv("SIZE_STOP_FACTOR_MAX", "1.20"))
            except Exception:
                factor_min, factor_max = 0.80, 1.20
            f_lo = min(factor_min, factor_max)
            f_hi = max(factor_min, factor_max)

            target_notional = float(equity) * float(size_pct) * leverage
            base_size = target_notional / float(entry_price)
            dist_pct = dist / float(entry_price)
            stop_factor = 1.0
            if stop_ref_pct > 0 and dist_pct > 0:
                stop_factor = math.sqrt(stop_ref_pct / dist_pct)
                stop_factor = max(f_lo, min(stop_factor, f_hi))
            raw_size = base_size * stop_factor

            default_cap_pct = min(1.0, max(float(size_pct), float(size_pct) * 2.0))
            try:
                pct_cap = float(os.getenv("PARITY_PCT_CAP", str(default_cap_pct)))
            except Exception:
                pct_cap = default_cap_pct

            max_size_by_cap = (float(equity) * float(pct_cap) * leverage) / float(entry_price) if entry_price else 0.0
            if max_size_by_cap > 0:
                raw_size = min(raw_size, max_size_by_cap)
            return max(0.0, float(raw_size))

        info = self._get_symbol_info(symbol)
        if info is None:
            logger.debug("[%s] Size descartado: symbol_info no disponible para %s", self.name, symbol)
            return None

        tick_value = _safe_float(getattr(info, "trade_tick_value", None))
        tick_size = _safe_float(getattr(info, "trade_tick_size", None))
        contract_size = _safe_float(getattr(info, "trade_contract_size", None))
        vol_min = _safe_float(getattr(info, "volume_min", None))
        vol_step = _safe_float(getattr(info, "volume_step", None))
        vol_max = _safe_float(getattr(info, "volume_max", None))

        if tick_value is None or tick_size is None or tick_value <= 0 or tick_size <= 0:
            logger.debug(
                "[%s] Size descartado: tick_value/tick_size inválidos (tick_value=%s tick_size=%s)",
                self.name,
                tick_value,
                tick_size,
            )
            return None
        if vol_min is None or vol_step is None or vol_min <= 0 or vol_step <= 0:
            logger.debug(
                "[%s] Size descartado: volumen min/step inválidos (vol_min=%s vol_step=%s)",
                self.name,
                vol_min,
                vol_step,
            )
            return None
        if vol_max is None or vol_max <= 0:
            vol_max = 100.0

        # Fallback clásico por riesgo al stop si no hay contract_size.
        value_per_price_unit = tick_value / tick_size  # dinero por 1.0 de precio por lote
        risk_amount = float(equity) * float(size_pct)
        raw_lots_by_stop = risk_amount / (dist * value_per_price_unit)

        raw_lots = raw_lots_by_stop
        capital_cap_lots: Optional[float] = None
        if contract_size is not None and contract_size > 0 and entry_price > 0:
            try:
                stop_ref_pct = float(os.getenv("SIZE_STOP_REFERENCE_PCT", "0.00075"))
            except Exception:
                stop_ref_pct = 0.00075
            try:
                factor_min = float(os.getenv("SIZE_STOP_FACTOR_MIN", "0.80"))
                factor_max = float(os.getenv("SIZE_STOP_FACTOR_MAX", "1.20"))
            except Exception:
                factor_min, factor_max = 0.80, 1.20
            f_lo = min(factor_min, factor_max)
            f_hi = max(factor_min, factor_max)

            target_notional = float(equity) * float(size_pct)
            base_lots = target_notional / (contract_size * float(entry_price))
            dist_pct = dist / float(entry_price)
            stop_factor = 1.0
            if stop_ref_pct > 0 and dist_pct > 0:
                stop_factor = math.sqrt(stop_ref_pct / dist_pct)
                stop_factor = max(f_lo, min(stop_factor, f_hi))
            raw_lots = base_lots * stop_factor

            default_cap_pct = min(1.0, max(float(size_pct), float(size_pct) * 2.0))
            cap_pct = default_cap_pct
            cap_pct_env = os.getenv("SIZE_NOTIONAL_CAP_PCT")
            if cap_pct_env:
                try:
                    cap_pct = float(cap_pct_env)
                except Exception:
                    cap_pct = default_cap_pct
            if os.getenv("PARITY_SIZING", "0") == "1":
                try:
                    cap_pct = float(os.getenv("PARITY_PCT_CAP", str(cap_pct)))
                except Exception:
                    pass
            if cap_pct > 0:
                capital_cap_lots = (float(equity) * float(cap_pct)) / (contract_size * float(entry_price))
                raw_lots = min(raw_lots, capital_cap_lots)

        # Encajar a step y límites
        if capital_cap_lots is not None and capital_cap_lots < vol_min:
            logger.debug(
                "[%s] Size descartado: cap nocional < volumen minimo (cap=%.6f vol_min=%.6f)",
                self.name,
                capital_cap_lots,
                vol_min,
            )
            return None

        lots = max(vol_min, min(raw_lots, vol_max))
        lots = math.floor(lots / vol_step) * vol_step
        lots = max(vol_min, lots)

        # Redondeo estable
        lots = float(round(lots, 8))
        if lots <= 0:
            logger.debug(
                "[%s] Size descartado: lotes <=0 tras ajuste (raw=%.6f adj=%.6f vol_min=%.6f vol_step=%.6f)",
                self.name,
                raw_lots,
                lots,
                vol_min,
                vol_step,
            )
            return None
        return lots

    def _update_trailing_stop(self, symbol: str, state: _TradeState) -> None:
        """Actualiza el stop activo según nuevos pivotes confirmados en TF_stop."""
        if not state.in_position:
            return
        if state.broken_zone is None:
            return
        if state.active_stop is None:
            return

        last_min = self._stop_last_min.get(symbol)
        last_max = self._stop_last_max.get(symbol)

        if state.direction > 0:
            if last_min is None:
                return
            broken_top = float(state.broken_zone["top"])
            if float(last_min) > broken_top and float(last_min) > float(state.active_stop):
                prev_stop = float(state.active_stop)
                new_stop = float(last_min)
                state.active_stop = new_stop
                state.pending_stop_price = new_stop
                state.stop_update_pending = True
                logger.info(
                    "[%s] TRAIL_STOP %s | dir=LONG prev_sl=%.5f new_sl=%.5f",
                    self.name,
                    symbol,
                    prev_stop,
                    new_stop,
                )
        else:
            if last_max is None:
                return
            broken_bot = float(state.broken_zone["bot"])
            if float(last_max) < broken_bot and float(last_max) < float(state.active_stop):
                prev_stop = float(state.active_stop)
                new_stop = float(last_max)
                state.active_stop = new_stop
                state.pending_stop_price = new_stop
                state.stop_update_pending = True
                logger.info(
                    "[%s] TRAIL_STOP %s | dir=SHORT prev_sl=%.5f new_sl=%.5f",
                    self.name,
                    symbol,
                    prev_stop,
                    new_stop,
                )

    # -----------------------------
    # Helpers de bracket TP/SL en broker
    # -----------------------------
    def _order_types_for_direction(self, direction: int) -> Tuple[str, str]:
        """Devuelve (tp_type, sl_type) según dirección."""
        if direction > 0:
            return ("SELL_LIMIT", "SELL_STOP")
        return ("BUY_LIMIT", "BUY_STOP")

    def _use_pending_brackets(self) -> bool:
        """Define si la estrategia debe gestionar SL/TP con órdenes pendientes opuestas.

        En cuentas MT5 tipo Hedge, las pendientes opuestas pueden abrir una nueva
        posición en lugar de cerrar la existente. Para evitar múltiples posiciones
        por símbolo en live, desactivamos este mecanismo para MetaTrader5Client
        y dejamos la protección en el SL/TP de la posición.
        """
        if os.getenv("FORCE_PENDING_BRACKETS", "0") == "1":
            return True
        if self.broker_client is None:
            return True
        return self.broker_client.__class__.__name__ != "MetaTrader5Client"

    def _cancel_bracket_orders(self, symbol: str, state: _TradeState) -> None:
        """Cancela órdenes pendientes TP/SL si existen."""
        if self.broker_client is None:
            return
        canceled_any = False
        for oid in (state.tp_order_id, state.sl_order_id):
            if oid is None:
                continue
            try:
                ok = self.broker_client.cancel_order(oid)  # type: ignore[attr-defined]
                canceled_any = canceled_any or bool(ok)
            except Exception as e:
                logger.warning("[%s] Error cancelando orden %s para %s: %s", self.name, oid, symbol, e)
        # Cancelar cualquier orden pendiente huérfana en broker
        try:
            if hasattr(self.broker_client, "get_open_orders"):
                magic = self._get_magic_number()
                for order in self.broker_client.get_open_orders(symbol=symbol, magic_number=magic):  # type: ignore[attr-defined]
                    oid = getattr(order, "order_id", None) or getattr(order, "ticket", None)
                    if oid:
                        try:
                            ok = self.broker_client.cancel_order(int(oid))  # type: ignore[attr-defined]
                            canceled_any = canceled_any or bool(ok)
                        except Exception:
                            continue
        except Exception:
            pass
        if canceled_any:
            logger.info(
                "[%s] BRACKET_CANCEL_OCO %s (tp_id=%s sl_id=%s)",
                self.name,
                symbol,
                state.tp_order_id,
                state.sl_order_id,
            )
        state.tp_order_id = None
        state.sl_order_id = None

    def _ensure_bracket_orders(
        self,
        symbol: str,
        state: _TradeState,
        position_volume: float,
    ) -> None:
        """Crea o recupera TP/SL pendientes en broker y aplica trailing pendiente."""
        if self.broker_client is None:
            return
        if state.active_tp is None or state.active_stop is None:
            return
        tp_type, sl_type = self._order_types_for_direction(state.direction)
        magic = self._get_magic_number()

        # Leer órdenes pendientes existentes (si el broker lo soporta)
        orders_by_type: Dict[str, Any] = {}
        try:
            if hasattr(self.broker_client, "get_open_orders"):
                open_orders = self.broker_client.get_open_orders(symbol=symbol, magic_number=magic)  # type: ignore[attr-defined]
                for o in open_orders or []:
                    o_type = str(getattr(o, "order_type", getattr(o, "type", "")))
                    orders_by_type[o_type] = o
        except Exception as e:
            logger.debug("[%s] No se pudieron leer órdenes pendientes: %s", self.name, e)

        def _order_id_from(order_obj: Any) -> Optional[int]:
            oid = getattr(order_obj, "order_id", None)
            if oid is None:
                oid = getattr(order_obj, "ticket", None)
            try:
                return int(oid) if oid is not None else None
            except Exception:
                return None

        # Recuperar si existen pero no están en estado local
        if state.tp_order_id is None and tp_type in orders_by_type:
            state.tp_order_id = _order_id_from(orders_by_type[tp_type])
            logger.info("[%s] BRACKET_RECOVER TP %s id=%s", self.name, symbol, state.tp_order_id)
        if state.sl_order_id is None and sl_type in orders_by_type:
            state.sl_order_id = _order_id_from(orders_by_type[sl_type])
            logger.info("[%s] BRACKET_RECOVER SL %s id=%s", self.name, symbol, state.sl_order_id)
        if state.tp_order_id is not None and tp_type not in orders_by_type:
            logger.info("[%s] BRACKET_RECOVER faltante TP para %s, recreando", self.name, symbol)
            state.tp_order_id = None
        if state.sl_order_id is not None and sl_type not in orders_by_type:
            logger.info("[%s] BRACKET_RECOVER faltante SL para %s, recreando", self.name, symbol)
            state.sl_order_id = None

        # Crear TP si falta
        created = False
        if state.tp_order_id is None:
            try:
                res_tp = self.broker_client.create_pending_order(  # type: ignore[attr-defined]
                    symbol=symbol,
                    order_type=tp_type,
                    volume=position_volume,
                    price=float(state.active_tp),
                    magic_number=magic,
                    comment=self.name,
                )
                if getattr(res_tp, "success", False):
                    state.tp_order_id = getattr(res_tp, "order_id", None)
                    created = True
            except Exception as e:
                logger.warning("[%s] Error creando TP pendiente para %s: %s", self.name, symbol, e)

        # Crear SL si falta
        if state.sl_order_id is None:
            try:
                res_sl = self.broker_client.create_pending_order(  # type: ignore[attr-defined]
                    symbol=symbol,
                    order_type=sl_type,
                    volume=position_volume,
                    price=float(state.active_stop),
                    magic_number=magic,
                    comment=self.name,
                )
                if getattr(res_sl, "success", False):
                    state.sl_order_id = getattr(res_sl, "order_id", None)
                    created = True
            except Exception as e:
                logger.warning("[%s] Error creando SL pendiente para %s: %s", self.name, symbol, e)

        if created:
            logger.info(
                "[%s] BRACKET_CREATE %s tp_id=%s sl_id=%s tp=%.5f sl=%.5f",
                self.name,
                symbol,
                state.tp_order_id,
                state.sl_order_id,
                float(state.active_tp),
                float(state.active_stop),
            )

        # Aplicar trailing pendiente (siguiente vela)
        if state.stop_update_pending and state.pending_stop_price is not None:
            self._update_stop_order(symbol, state, position_volume)

    def _update_stop_order(self, symbol: str, state: _TradeState, position_volume: float) -> None:
        """Actualiza la orden de SL pendiente en broker (modify o cancel+create)."""
        if self.broker_client is None:
            return
        if state.pending_stop_price is None:
            return
        if not self._use_pending_brackets():
            # En MT5 real (hedge), no usar pendientes opuestas como SL para evitar
            # abrir posiciones adicionales. Si el broker soporta SLTP sobre posición,
            # actualizar el SL directamente en la posición.
            if hasattr(self.broker_client, "modify_position_sl_tp"):
                try:
                    res = self.broker_client.modify_position_sl_tp(  # type: ignore[attr-defined]
                        symbol=symbol,
                        magic_number=self._get_magic_number(),
                        stop_loss=float(state.pending_stop_price),
                        take_profit=float(state.active_tp) if state.active_tp is not None else None,
                    )
                    if getattr(res, "success", False):
                        logger.info(
                            "[%s] POSITION_SL_UPDATE %s new_stop=%.5f",
                            self.name,
                            symbol,
                            float(state.pending_stop_price),
                        )
                        state.active_stop = float(state.pending_stop_price)
                        state.stop_update_pending = False
                        state.pending_stop_price = None
                        return
                except Exception as e:
                    logger.warning("[%s] Error actualizando SL de posición en %s: %s", self.name, symbol, e)
            return
        tp_type, sl_type = self._order_types_for_direction(state.direction)
        magic = self._get_magic_number()
        new_price = float(state.pending_stop_price)

        updated = False
        if state.sl_order_id is not None and hasattr(self.broker_client, "modify_order"):
            try:
                res = self.broker_client.modify_order(state.sl_order_id, price=new_price)  # type: ignore[attr-defined]
                updated = getattr(res, "success", False)
            except Exception as e:
                logger.debug("[%s] No se pudo modificar SL pendiente %s: %s", self.name, state.sl_order_id, e)

        if not updated and state.sl_order_id is not None:
            try:
                self.broker_client.cancel_order(state.sl_order_id)  # type: ignore[attr-defined]
            except Exception:
                pass
            state.sl_order_id = None

        if not updated:
            try:
                res_new = self.broker_client.create_pending_order(  # type: ignore[attr-defined]
                    symbol=symbol,
                    order_type=sl_type,
                    volume=position_volume,
                    price=new_price,
                    magic_number=magic,
                    comment=self.name,
                )
                updated = getattr(res_new, "success", False)
                if updated:
                    state.sl_order_id = getattr(res_new, "order_id", None)
            except Exception as e:
                logger.warning("[%s] Error recreando SL pendiente para %s: %s", self.name, symbol, e)

        if updated:
            logger.info(
                "[%s] BRACKET_UPDATE_SL %s new_stop=%.5f sl_id=%s",
                self.name,
                symbol,
                new_price,
                state.sl_order_id,
            )
            # Paridad/dev: sincronizar stop_loss en FakeBroker para que los cierres SL/TP
            # usen el trailing actualizado (en MT5 real esto lo gestiona el broker).
            try:
                if hasattr(self.broker_client, "open_positions"):
                    magic = self._get_magic_number()
                    for pos in getattr(self.broker_client, "open_positions", []):
                        if getattr(pos, "symbol", None) == symbol and getattr(pos, "magic_number", None) == magic:
                            setattr(pos, "stop_loss", float(new_price))
            except Exception:
                pass
            state.stop_update_pending = False
            state.pending_stop_price = None

    # -----------------------------
    # API principal
    # -----------------------------
    def generate_signals(self, data_by_timeframe: Dict[str, pd.DataFrame]) -> List[Signal]:
        signals: List[Signal] = []

        ok, reason = self._validate_timeframes(data_by_timeframe)
        if not ok:
            logger.debug("[%s] %s", self.name, reason)
            return signals

        symbol = self._get_symbol(data_by_timeframe)
        if not symbol:
            logger.debug("[%s] No se pudo determinar symbol desde df.attrs", self.name)
            return signals

        if self.allowed_symbols is not None and symbol not in self.allowed_symbols:
            return signals

        params = self._resolve_params(symbol)

        # Preparar series cerradas
        df_entry_raw = data_by_timeframe[self.tf_entry]
        # Usamos todas las velas cerradas tal como llegan del proveedor (ya son closes).
        # Evitamos descartar sistemáticamente la última fila para alinear con Backtrader, que consume la vela cerrada actual.
        df_entry = df_entry_raw
        df_zone = self._closed_series(data_by_timeframe[self.tf_zone])
        df_stop = self._closed_series(data_by_timeframe[self.tf_stop])

        # Validar mínimos de datos
        if df_entry is None or df_zone is None or df_stop is None:
            return signals
        if df_entry.empty or df_zone.empty or df_stop.empty:
            return signals

        self._ensure_symbol_state(symbol)

        # 1) Actualizar zonas (TF_zone) y pivotes stop (TF_stop)
        self._process_zone_tf(symbol, df_zone, params)
        self._process_stop_tf(symbol, df_stop)

        zones_count = len(self._saved_zones_by_symbol.get(symbol, []))
        logger.debug(
            "[%s] Estado %s | entry=%d zone=%d stop=%d | zonas_guardadas=%d",
            self.name,
            symbol,
            len(df_entry),
            len(df_zone),
            len(df_stop),
            zones_count,
        )

        state = self._trade_state[symbol]
        open_pos = self._get_open_position(symbol)

        # 2) Sincronizar estado de "posición" desde broker (si hay broker_client)
        if open_pos is not None:
            state.in_position = True
            open_dir = int(getattr(open_pos, "direction", 0) or 0)
            is_parity_reversal = bool(getattr(open_pos, "_parity_reversal", False))
            if is_parity_reversal or (open_dir != 0 and state.direction != 0 and open_dir != state.direction):
                # Reversión implícita detectada en broker (p. ej. doble fill SL/TP en misma vela):
                # reiniciar estado interno para no arrastrar brackets/trailing de la dirección previa.
                state = _TradeState(in_position=True, direction=open_dir)
                self._trade_state[symbol] = state
            # Recuperar SL/TP y dirección si el estado estaba limpio
            if state.active_stop is None and getattr(open_pos, "stop_loss", None) is not None:
                state.active_stop = float(open_pos.stop_loss)
            if state.active_tp is None and getattr(open_pos, "take_profit", None) is not None:
                state.active_tp = float(open_pos.take_profit)
            if state.direction == 0 and state.active_tp is not None and state.active_stop is not None:
                state.direction = 1 if float(state.active_tp) > float(state.active_stop) else -1
        else:
            # si no hay posición real, limpiar estado local para mantener consistencia
            if state.in_position:
                self._cancel_bracket_orders(symbol, state)
                state = _TradeState()
                self._trade_state[symbol] = state

        # En MT5 real (hedge) no usamos brackets pendientes opuestos.
        # Limpiamos cualquier remanente pendiente del símbolo+magic en cada ciclo,
        # incluso si no hay posición abierta.
        if not self._use_pending_brackets():
            self._cancel_bracket_orders(symbol, state)

        # 3) Si hay posición: solo gestionar brackets; no generar nuevas señales
        if state.in_position:
            # Regla actual: sin trailing; solo salida por SL inicial o TP objetivo.
            state.stop_update_pending = False
            state.pending_stop_price = None
            pos_volume = float(getattr(open_pos, "volume", 0.0)) if open_pos is not None else 0.0
            if pos_volume > 0 and self._use_pending_brackets():
                self._ensure_bracket_orders(symbol, state, pos_volume)
            elif pos_volume > 0:
                # Limpieza de pendientes heredadas para evitar sobre-apertura en cuentas hedge.
                self._cancel_bracket_orders(symbol, state)
                if state.stop_update_pending and state.pending_stop_price is not None:
                    self._update_stop_order(symbol, state, pos_volume)
            else:
                logger.debug("[%s] Sin volumen de posición para gestionar bracket en %s", self.name, symbol)
            return signals

        # 4) Si no hay posición: buscar entrada por ruptura
        zones = self._saved_zones_by_symbol.get(symbol, [])
        breakout = self._check_breakout(df_entry, zones, params.p)
        if breakout is None:
            return signals

        broken_zone, direction = breakout
        target_zone = self._select_target_zone(zones, broken_zone, direction)
        if target_zone is None:
            return signals

        # Entry price de referencia: close de la vela actual cerrada en TF_entry
        entry_price = float(df_entry.iloc[-1]["close"])

        # Stop inicial basado en pivotes TF_stop confirmados
        stop_price = self._find_stop_inside_broken_zone(symbol, broken_zone, direction)
        if stop_price is None:
            logger.debug(
                "[%s] Señal descartada: sin pivote stop dentro de zona rota (%s dir=%s)",
                self.name,
                symbol,
                "LONG" if direction > 0 else "SHORT",
            )
            return signals
        stop_price_f = float(stop_price)

        # TP: borde más cercano de la zona objetivo
        tp_price = float(self._tp_edge_nearest(target_zone, entry_price))

        skip_sltp_validation = os.getenv("PARITY_SKIP_SLTP_VALIDATION", "0") == "1"
        # Validaciones coherentes (se pueden omitir en modo paridad)
        if not skip_sltp_validation:
            if direction > 0:
                if not (stop_price_f < entry_price < tp_price):
                    logger.debug("[%s] SL/TP inválidos para LONG %s: entry=%.5f sl=%.5f tp=%.5f",
                                 self.name, symbol, entry_price, stop_price_f, tp_price)
                    return signals
            else:
                if not (tp_price < entry_price < stop_price_f):
                    logger.debug("[%s] SL/TP inválidos para SHORT %s: entry=%.5f sl=%.5f tp=%.5f",
                                 self.name, symbol, entry_price, stop_price_f, tp_price)
                    return signals

        # Lot size por riesgo (requiere equity + symbol_info)
        lot_size = self._calc_lot_size_by_stop(symbol, entry_price, stop_price_f, params.size_pct)
        if lot_size is None:
            logger.debug("[%s] No se pudo calcular lot_size (falta equity/symbol_info)", self.name)
            return signals

        ts_event = _format_ts(df_entry.index[-1])
        bar_idx = len(df_entry) - 1
        self._emit_event(
            {
                "ts_event": ts_event,
                "bar_index": bar_idx,
                "symbol": symbol,
                "timeframe": self.tf_entry,
                "event_type": "signal",
                "side": "long" if direction > 0 else "short",
                "price_ref": "close",
                "price": entry_price,
                "size": float(lot_size),
                "commission": 0.0,
                "slippage": 0.0,
                "reason": "breakout_long" if direction > 0 else "breakout_short",
                "order_id": "",
            }
        )

        sig_type = SignalType.BUY if direction > 0 else SignalType.SELL
        signals.append(
            Signal(
                symbol=symbol,
                strategy_name=self.name,
                timeframe=self.tf_entry,
                signal_type=sig_type,
                size=float(lot_size),
                stop_loss=float(stop_price_f),
                take_profit=float(tp_price),
            )
        )

        # Congelar estado para gestión interna desde el siguiente ciclo
        state.in_position = True  # asumimos éxito; se re-sincroniza con broker en el siguiente ciclo
        state.direction = int(direction)
        state.entry_timeframe = self.tf_entry
        state.broken_zone = dict(broken_zone)
        state.target_zone = dict(target_zone)
        state.active_stop = float(stop_price_f)
        state.active_tp = float(tp_price)
        state.tp_order_id = None
        state.sl_order_id = None
        state.stop_update_pending = False
        state.pending_stop_price = None

        logger.info(
            "[%s] ENTRADA_SIGNAL %s | entry=%.5f size=%.4f SL=%.5f TP=%.5f | broken=[%.5f, %.5f] target=[%.5f, %.5f]",
            self.name,
            symbol,
            entry_price,
            float(lot_size),
            float(stop_price_f),
            float(tp_price),
            float(broken_zone["bot"]),
            float(broken_zone["top"]),
            float(target_zone["bot"]),
            float(target_zone["top"]),
        )
        return signals
