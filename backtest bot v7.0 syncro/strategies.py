# ================================================================
# strategies.py â€” Estrategias Backtrader (refactor suave)
# ================================================================
# Cambios respecto a tu versiÃ³n anterior:
#   - Eliminada la funciÃ³n sync_multi_asset_data (ya estÃ¡ en DataManager).
#   - Mantenida max_min_search (idÃ©ntica, Ãºtil para pivotes).
#   - _calc_size se mueve a _BaseLoggedStrategy (reutilizable),
#     sin cambiar la forma de llamarla desde las estrategias.
#   - LÃ³gica de las estrategias: EXACTAMENTE igual.
#   - Comentarios extensos y logs para depurar paso a paso.()
#   
# ================================================================

from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field
import numpy as np
import backtrader as bt
import math
import statistics
from typing import Optional, Tuple, Dict, List
import os
from pathlib import Path
from event_logger import EventLogger
try:
    import talib  # type: ignore
except Exception:
    talib = None

class size_constructor:
    """
    Mixin de utilidades de sizing para Backtrader.
    Depende de:
      - self.broker           â†’ broker de Backtrader (getvalue, getcommissioninfo)
      - self.data             â†’ data principal si no se pasa `data` explÃ­cito
      - self.getposition(d)   â†’ posiciÃ³n para un `data` (multi-data). Si no existe, usa self.position

    MÃ©todos:
      - size_margin(data=None, fallback_margin=1.0) -> (leverage, margin)
      - size_fractal(size, data=None, lot_step=None, min_size=0.0) -> size_ajustada
      - size_data(data=None) -> {'price', 'equity', 'current'}
    """

    # ------------------- Helpers de sizing -------------------

    def size_margin(self, data=None, fallback_margin: float = 1.0) -> Tuple[float, float]:
        """
        Devuelve (leverage, margin) a partir del CommInfo del broker para `data`.
        - Si no hay margin definido, usa `fallback_margin` (1.0 por defecto) â†’ leverage=1.0.
        """
        data = data or self.data
        comm_info = self.broker.getcommissioninfo(data)
        # CommInfo estÃ¡ndar expone parÃ¡metros en comm_info.p (si existe)
        if hasattr(comm_info, 'p'):
            margin = getattr(comm_info.p, 'margin', None)
        else:
            margin = None

        if margin is None or margin <= 0:
            margin = fallback_margin

        leverage = 1.0 / margin if margin and margin > 0 else 1.0
        if os.getenv("PARITY_SIZING", "0") == "1":
            try:
                lev_env = float(os.getenv("PARITY_LEVERAGE", "0") or 0)
                if lev_env > 0:
                    leverage = lev_env
                    margin = 1.0 / lev_env
            except Exception:
                pass
        return leverage, margin

    def size_fractal(
        self,
        size: float,
        data=None,
        lot_step: Optional[float] = None,
        min_size: float = 0.0
    ) -> float:
        """
        Ajusta `size` al paso mÃ­nimo permitido (si existe), permitiendo tamaÃ±os < 1.
        - Mantiene fraccionales por defecto (no redondea a entero).
        - Si `lot_step` no se pasa, intenta detectarlo de CommInfo (p.lot_step o p.size_step).
        - `min_size` fuerza un mÃ­nimo absoluto si el broker lo requiere.
        """
        data = data or self.data

        # Intentar descubrir un paso de lote desde CommInfo
        if lot_step is None:
            ci = self.broker.getcommissioninfo(data)
            if hasattr(ci, 'p'):
                lot_step = getattr(ci.p, 'lot_step', None) or getattr(ci.p, 'size_step', None)

        s = float(size)

        # Encaje al paso mÃ­nimo si existe
        if lot_step and lot_step > 0:
            # nÃºmero de pasos completos
            steps = math.floor(abs(s) / lot_step)
            units = steps * lot_step
            s = math.copysign(units, s)

        # Enforce tamaÃ±o mÃ­nimo (si procede)
        if 0.0 < abs(s) < min_size:
            s = math.copysign(min_size, s)

        return s

    def size_data(self, data=None) -> Dict[str, float]:
        """
        Lecturas internas para sizing:
          - price: Ãºltimo close
          - equity: valor total de la cuenta
          - current: tamaÃ±o de la posiciÃ³n actual en `data`
        Compatible con multi-data.
        """
        data = data or self.data
        price = float(data.close[0])
        equity = float(self.broker.getvalue())
        if os.getenv("PARITY_SIZING", "0") == "1":
            try:
                eq_env = float(os.getenv("PARITY_EQUITY", "") or 0)
                if eq_env > 0:
                    equity = eq_env
            except Exception:
                pass

        # Multi-data: getposition(data) si existe; si no, self.position
        if hasattr(self, 'getposition'):
            pos = self.getposition(data)
        else:
            pos = getattr(self, 'position', None)

        current = float(getattr(pos, 'size', 0.0)) if pos is not None else 0.0
        return {'price': price, 'equity': equity, 'current': current}


class Pivot3Candle(bt.Indicator):
    """
    Pivotes 3 velas (sin look-ahead):
      - MÃ¡x local confirmado si: high[-3] < high[-2] > high[-1]
      - MÃ­n local confirmado si:  low [-3] >  low [-2] < low [-1]
    Marca pivotes puntuales y mantiene la "memoria" del Ãºltimo pivote
    en las lÃ­neas last_pivot_* y en los atributos last_max/last_min.
    """
    lines = ('pivot_max', 'pivot_min', 'last_pivot_max', 'last_pivot_min')
    plotinfo = dict(subplot=False)
    plotlines = dict(
        pivot_max=dict(_name='Maximo Local', marker='v', markersize=8.0, color='black', fillstyle='full', ls=''),
        pivot_min=dict(_name='Minimo Local', marker='^', markersize=8.0, color='black', fillstyle='full', ls=''),
        last_pivot_max=dict(_name='PIVOT_LAST_MAX', ls='--'),
        last_pivot_min=dict(_name='PIVOT_LAST_MIN', ls='--'),
    )

    def __init__(self):
        self.addminperiod(3)
        # estado persistente (se vuelca en cada vela; no dependemos de [-1])
        self._lpmax = float('nan')
        self._lpmin = float('nan')
        # accesibles desde estrategias
        self.last_max = None
        self.last_min = None

    def next(self):
        
        # 1) detectar pivote central en (-2)
        h3 = float(self.data.high[-3]); h2 = float(self.data.high[-2]); h1 = float(self.data.high[-1])
        l3 = float(self.data.low[-3]);  l2 = float(self.data.low[-2]);  l1 = float(self.data.low[-1])

        # Detectar si hay pivote en esta vela
        has_pivot_max = False
        has_pivot_min = False

        if (h2 > h3) and (h2 > h1):
            self.lines.pivot_max[0] = h2
            self._lpmax = h2  # actualizar memoria
            has_pivot_max = True
        else:
            self.lines.pivot_max[0] = float('nan')

        if (l2 < l3) and (l2 < l1):
            self.lines.pivot_min[0] = l2
            self._lpmin = l2  # actualizar memoria
            has_pivot_min = True
        else:
            self.lines.pivot_min[0] = float('nan')

        # 2) Persistencia condicional para mÃ¡ximos
        if has_pivot_max:
            # Si hay pivote, siempre actualizar
            self._lpmax = h2
        else:
            # Si no hay pivote, actualizar solo si el high actual es mayor que last_max
            current_high = float(self.data.high[0])
            if not math.isnan(self._lpmax):
                if current_high > self._lpmax:
                    self._lpmax = current_high
            else:
                # Si no hay last_max previo, inicializar con el high actual
                self._lpmax = current_high

        # 3) Persistencia condicional para mÃ­nimos
        if has_pivot_min:
            # Si hay pivote, siempre actualizar
            self._lpmin = l2
        else:
            # Si no hay pivote, actualizar solo si el low actual es menor que last_min
            current_low = float(self.data.low[0])
            if not math.isnan(self._lpmin):
                if current_low < self._lpmin:
                    self._lpmin = current_low
            else:
                # Si no hay last_min previo, inicializar con el low actual
                self._lpmin = current_low

        # 4) volcar memoria a las lÃ­neas de "Ãºltimo pivote" (se plotean siempre)
        self.lines.last_pivot_max[0] = self._lpmax
        self.lines.last_pivot_min[0] = self._lpmin

        # 5) sincronizar atributos para usarlos como stop en estrategias
        if not math.isnan(self._lpmax):
            self.last_max = self._lpmax
        if not math.isnan(self._lpmin):
            self.last_min = self._lpmin

        return self.last_max, self.last_min
    
    @dataclass
    class _Zone:
        mid: float
        half_width: float
        touches: int = 1
        pivots: list = field(default_factory=list)
        last_touch_bar: int = 0

        def top(self) -> float:
            return self.mid + self.half_width

        def bot(self) -> float:
            return self.mid - self.half_width

class PivotZone(bt.Indicator):
    """
    ============================
    INDICADOR: ZONA PIVOTE
    ============================

    CONCEPTO (definiciÃ³n estricta):
    Una ZONA PIVOTE vÃ¡lida existe Ãºnicamente cuando se cumplen AMBAS condiciones:

    1) AGRUPACIÃ“N:
       Existen al menos `n3` pivotes (mÃ¡ximos o mÃ­nimos) concentrados
       dentro de un rango de precios definido como:
           width = ATR(n1) * (n2 / 100)

    2) CAMBIO DE ROL (PULLBACK):
       El precio ha estado claramente por ENCIMA y por DEBAJO de la MISMA zona.
       Esto demuestra memoria de mercado:
           soporte -> resistencia  o  resistencia -> soporte

    Hasta que ambas condiciones se cumplen:
    - NO existe zona pivote
    - Solo existe un PROYECTO DE ZONA

    Cuando ambas condiciones se cumplen:
    - La zona se BLOQUEA
    - No se reemplaza nunca
    - Su centro y su ancho quedan fijos para siempre
    """

    # ======================================================
    # LÃ­neas del indicador (salida observable)
    # ======================================================
    lines = (
        'mid',      # Centro de la zona pivote (precio representativo)
        'top',      # LÃ­mite superior de la zona
        'bot',      # LÃ­mite inferior de la zona
        'touches',  # NÃºmero total de pivotes que han reforzado la zona
        'valid',    # 1 = zona pivote vÃ¡lida | 0 = no vÃ¡lida (solo proyecto)
        'role',     # +1 soporte | -1 resistencia | 0 precio dentro de la zona
    )

    # ======================================================
    # ParÃ¡metros (optimizables en Backtrader)
    # ======================================================
    params = dict(
        n1=14,   # Compatibilidad de firma; ATR se fija internamente en 14
        n2=60,   # Ancho de la zona como % del ATR (60 = 0.60 * ATR)
        n3=3,    # NÂº mÃ­nimo de pivotes para iniciar un proyecto de zona
    )

    plotinfo = dict(subplot=False)
    plotlines = dict(
        mid=dict(ls='-'),
        top=dict(ls='--'),
        bot=dict(ls='--'),
    )

    def __init__(self):
        # ----------------------------------------------
        # Detector de pivotes (3 velas, ya existente)
        # ----------------------------------------------
        self.piv = Pivot3Candle(self.data)
        # Pivote calculado desde data (paridad con bot).
        self._last_pivot_price: Optional[float] = None
        self._last_pivot_reason: str = ""
        # Guard para evitar recalculo en barras repetidas (multi-TF).
        self._last_len: int = 0

        # ----------------------------------------------
        # ATR para definir el ancho dinÃ¡mico de la zona
        # ----------------------------------------------
        # IMPORTANTE (paridad con bot):
        # - En el bot, ATR (TA-Lib) devuelve NaN hasta disponer de n1+1 barras.
        # - En Backtrader, ATR tambiÃ©n requiere n1+1 por el uso de close(-1).
        # Para alinear la disponibilidad del ancho sin forzar minperiod extra,
        # calculamos ATR manualmente y usamos fallback cuando aÃºn no hay suficiente
        # historial. AsÃ­ podemos empezar en len == n1 (bar_index n1-1) con fallback.
        self._atr_period = 14
        self.addminperiod(max(3, self._atr_period))

        # ======================================================
        # Estado interno (no visible externamente)
        # ======================================================

        # Lista de TODOS los pivotes que refuerzan el proyecto/zona
        self._pz_prices = []

        # Centro provisional del proyecto de zona
        self._pz_mid = float('nan')

        # Flags de validaciÃ³n del pullback (cambio de rol)
        self._has_been_above = False  # el precio ha cerrado por encima de la zona
        self._has_been_below = False  # el precio ha cerrado por debajo de la zona

        # Estado de bloqueo
        self._locked = False          # True cuando la zona es pivote vÃ¡lido
        self._locked_mid = float('nan')
        self._locked_width = float('nan')

    # ======================================================
    # Helpers internos
    # ======================================================
    def _series_to_list(self, line) -> List[float]:
        """
        Devuelve los valores del line en orden cronolÃ³gico (oldest -> newest).
        Backtrader entrega get(size=N) en orden cronolÃ³gico, por lo que
        no se invierte (evita falsos positivos cuando el precio se repite).
        """
        size = len(self.data)
        if size <= 0:
            return []
        values = list(line.get(size=size))
        if not values:
            return []
        return [float(v) for v in values]

    def _calc_atr_talib(self) -> Optional[float]:
        """
        Calcula ATR con TA-Lib (misma semÃ¡ntica que el bot).
        - Devuelve None si TA-Lib no estÃ¡ disponible o no hay barras suficientes.
        - TA-Lib empieza a devolver ATR vÃ¡lido a partir de n1+1 barras.
        """
        if talib is None:
            return None
        n1 = int(self._atr_period)
        total = len(self.data)
        if total < (n1 + 1):
            return None

        highs = self._series_to_list(self.data.high)
        lows = self._series_to_list(self.data.low)
        closes = self._series_to_list(self.data.close)
        if not highs or not lows or not closes:
            return None

        atr_arr = talib.ATR(
            np.asarray(highs, dtype=float),
            np.asarray(lows, dtype=float),
            np.asarray(closes, dtype=float),
            timeperiod=n1,
        )
        if atr_arr is None or len(atr_arr) == 0:
            return None
        atr_val = float(atr_arr[-1])
        if not math.isfinite(atr_val):
            return None
        return atr_val

    def _calc_width(self) -> float:
        """
        Calcula el ancho de la zona como porcentaje del ATR.
        n2 se divide entre 100 para permitir optimizaciÃ³n entera.
        """
        atr = self._calc_atr_talib()
        width = float("nan")
        if atr is not None and math.isfinite(float(atr)):
            width = float(atr) * (self.p.n2 / 100.0)

        # Fallback de seguridad (evita NaN o width=0)
        if not math.isfinite(width) or width <= 0:
            width = abs(float(self.data.close[0])) * 0.001  # 0.1%

        return float(width)

    def _recalc_mid(self):
        """
        Recalcula el centro de la zona usando TODOS los pivotes acumulados.
        No hay lÃ­mite: la memoria del mercado es completa.
        """
        if not self._pz_prices:
            self._pz_mid = float('nan')
            return

        # Mediana: robusta ante outliers
        self._pz_mid = float(statistics.median(self._pz_prices))

    def reset(self):
        """
        Resetea el indicador para buscar una nueva zona pivote.
        Llamar desde la estrategia despues de guardar la zona validada.
        """
        self._pz_prices = []
        self._pz_mid = float('nan')
        self._has_been_above = False
        self._has_been_below = False
        self._locked = False
        self._locked_mid = float('nan')
        self._locked_width = float('nan')
        print(f"[PIVOT ZONE RESET] Buscando nueva zona...")

    # ======================================================
    # LÃ³gica principal
    # ======================================================
    def next(self):
        # Si la zona ya es vÃ¡lida, usamos su ancho bloqueado
        if len(self.data) == self._last_len:
            return
        self._last_len = len(self.data)
        width = self._locked_width if self._locked else self._calc_width()

        # --------------------------------------------------
        # 1) Detectar pivote confirmado en esta barra
        # --------------------------------------------------
        self._last_pivot_price = None
        self._last_pivot_reason = ""
        pivot_price = float('nan')
        if len(self.data) >= 4:
            h3 = float(self.data.high[-3])
            h2 = float(self.data.high[-2])
            h1 = float(self.data.high[-1])
            l3 = float(self.data.low[-3])
            l2 = float(self.data.low[-2])
            l1 = float(self.data.low[-1])
            if (h2 > h3) and (h2 > h1):
                pivot_price = h2
                self._last_pivot_price = h2
                self._last_pivot_reason = "pivot_max"
            elif (l2 < l3) and (l2 < l1):
                pivot_price = l2
                self._last_pivot_price = l2
                self._last_pivot_reason = "pivot_min"

        # --------------------------------------------------
        # 2) ConstrucciÃ³n del PROYECTO DE ZONA
        #    (solo si aÃºn no estÃ¡ bloqueada)
        # --------------------------------------------------
        if math.isfinite(pivot_price) and not self._locked:
            if not self._pz_prices:
                # Primer pivote â†’ nace el proyecto
                self._pz_prices = [pivot_price]
                self._recalc_mid()
            else:
                # Â¿El pivote cae dentro del rango actual?
                if abs(pivot_price - self._pz_mid) <= width:
                    self._pz_prices.append(pivot_price)
                    self._recalc_mid()
                else:
                    # Pivot fuera â†’ nuevo proyecto de zona
                    self._pz_prices = [pivot_price]
                    self._recalc_mid()

        # --------------------------------------------------
        # 3) Seguimiento del precio respecto a la zona
        #    (validaciÃ³n del cambio de rol)
        # --------------------------------------------------
        mid = self._pz_mid
        if math.isfinite(mid):
            top = mid + width
            bot = mid - width
            close = float(self.data.close[0])

            # Cierres claros fuera de la zona
            if close > top:
                self._has_been_above = True
            elif close < bot:
                self._has_been_below = True

        # --------------------------------------------------
        # 4) VALIDACIÃ“N FINAL DE ZONA PIVOTE
        # --------------------------------------------------
        if (
            not self._locked
            and len(self._pz_prices) >= int(self.p.n3)
            and self._has_been_above
            and self._has_been_below
        ):
            # El mercado ha demostrado memoria estructural
            self._locked = True
            self._locked_mid = float(self._pz_mid)
            self._locked_width = float(width)

        # --------------------------------------------------
        # 5) PublicaciÃ³n de lÃ­neas del indicador
        # --------------------------------------------------
        mid = self._locked_mid if self._locked else self._pz_mid

        if math.isfinite(mid):
            top = mid + width
            bot = mid - width
            close = float(self.data.close[0])

            # Rol dinÃ¡mico de la zona
            if close > top:
                role = 1.0      # soporte
            elif close < bot:
                role = -1.0     # resistencia
            else:
                role = 0.0      # precio dentro de la zona

            self.lines.mid[0] = mid
            self.lines.top[0] = top
            self.lines.bot[0] = bot
            self.lines.touches[0] = float(len(self._pz_prices))
            self.lines.valid[0] = 1.0 if self._locked else 0.0
            self.lines.role[0] = role
        else:
            nan = float('nan')
            self.lines.mid[0] = nan
            self.lines.top[0] = nan
            self.lines.bot[0] = nan
            self.lines.touches[0] = 0.0
            self.lines.valid[0] = 0.0
            self.lines.role[0] = 0.0

# ================================================================
# BASE: estrategia con logs y utilidades comunes
# ================================================================
class _BaseLoggedStrategy(bt.Strategy, size_constructor):
    """
    Clase base para estrategias:
      - Registra Ã³rdenes y trades cerrados (para depurar/exportar).
      - Expone lÃ­neas 'pivot_max' y 'pivot_min' para plotear pivotes.
    """

    def __init__(self):


        # Acumuladores de eventos (para auditorÃ­a)
        self._orders_log: List[Dict[str, Any]] = []  # Todas las Ã³rdenes completadas
        self._closed_trades: List[Dict[str, Any]] = []  # Trades cerrados reconstruidos
        # Metadatos opcionales para alinear fills con el bar de envÃ­o (paridad bot).
        self._parity_order_meta: Dict[int, Dict[str, Any]] = {}

    # AÃ±adir dentro de _BaseLoggedStrategy
    def size_percent(
        self,
        pct: float,
        data=None,
        *,
        lot_step: Optional[float] = None,
        min_size: float = 0.0,
        fallback_margin: float = 1.0
    ) -> float:
        """
        Calcula el nÂº de unidades para que la posiciÃ³n equivalga a `pct` del equity actual.
        - No abre/cierra posiciones.
        - No usa la posiciÃ³n actual ni aplica direcciÃ³n.
        - Devuelve SIEMPRE un tamaÃ±o positivo (float) que luego usarÃ¡s en buy()/sell().
        Ej.: pct=0.05 â†’ 5% del equity. Si pct<0, se toma abs(pct).

        Params:
        pct: fracciÃ³n del equity objetivo (0.05 = 5%).
        data: feed (por defecto self.data).
        lot_step: paso mÃ­nimo de lote; si None intenta detectarlo desde CommInfo.
        min_size: tamaÃ±o mÃ­nimo absoluto (si el broker lo exige).
        fallback_margin: margen por defecto si CommInfo no define margin.

        Return:
        float >= 0.0 con el tamaÃ±o (unidades).
        """
        data = data or self.data

        # Lecturas internas
        sd = self.size_data(data)  # {'price','equity','current'} -> aquÃ­ ignoramos 'current'
        price = sd['price']
        equity = sd['equity']

        if price <= 0.0 or equity <= 0.0 or pct == 0.0:
            return 0.0

        # Apalancamiento efectivo del broker
        leverage, _ = self.size_margin(data, fallback_margin=fallback_margin)

        # TamaÃ±o bruto (sin direcciÃ³n): valor objetivo / precio
        target_value = equity * abs(pct) * leverage
        size = target_value / price

        # Encajar a pasos y mÃ­nimos permitidos (tamaÃ±os fraccionales OK)
        size = self.size_fractal(size, data=data, lot_step=lot_step, min_size=min_size)

        return max(0.0, float(size))

    def size_percent_by_stop(
        self,
        risk_pct: float,
        stop_price: float,
        *,
        data=None,
        cushion: float = 0.0,
        mult: float = 1.0,
        pct_cap: float = 0.50
    ) -> float:
        """
        Dimensiona con `risk_pct` como nocional objetivo y ajuste suave por stop.

        PolÃ­tica:
        - Base: nocional objetivo ~= equity * risk_pct * leverage.
        - ModulaciÃ³n por stop: tamaÃ±o sube/baja suavemente segÃºn distancia relativa al stop.
        - Cap nocional superior para limitar exposiciones extremas.
        """
        data = data or self.data
        sd = self.size_data(data)
        price = sd['price']
        equity = sd['equity']
        
        if risk_pct <= 0 or stop_price is None or price <= 0 or equity <= 0:
            return 0.0

        # Distancia efectiva al stop
        D = abs(price - float(stop_price))
        if D <= 0:
            return 0.0
        if cushion > 0:
            D *= (1.0 + float(cushion))

        # Obtener leverage (ANTES de usarlo).
        leverage, _ = self.size_margin(data)
        leverage = max(1.0, float(leverage))

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

        target_notional = equity * abs(risk_pct) * leverage
        base_size = target_notional / price
        dist_pct = D / price
        stop_factor = 1.0
        if stop_ref_pct > 0 and dist_pct > 0:
            stop_factor = math.sqrt(stop_ref_pct / dist_pct)
            stop_factor = max(f_lo, min(stop_factor, f_hi))
        size = base_size * stop_factor

        default_cap_pct = min(1.0, max(abs(risk_pct), abs(risk_pct) * 2.0))
        cap_pct = default_cap_pct if default_cap_pct > 0 else float(pct_cap)
        if os.getenv("PARITY_SIZING", "0") == "1":
            try:
                pct_env = float(os.getenv("PARITY_PCT_CAP", "") or 0)
                if pct_env > 0:
                    cap_pct = pct_env
            except Exception:
                pass
        else:
            cap_env = os.getenv("SIZE_NOTIONAL_CAP_PCT")
            if cap_env:
                try:
                    cap_pct = float(cap_env)
                except Exception:
                    pass

        max_size_by_cap = (equity * float(cap_pct) * leverage) / price if leverage > 0 else 0.0
        if max_size_by_cap > 0:
            size = min(size, max_size_by_cap)

        # Ajustar a pasos y mÃ­nimos permitidos.
        size = self.size_fractal(size, data=data, lot_step=None, min_size=0.0)
        
        return max(0.0, float(size))

    def _export_indicators_helper(self, *indicators, names=None):
        """
        Helper para exportar indicadores de Backtrader a diccionario.
        Convierte las lÃ­neas de indicadores en listas serializables para plotting/export.

        Uso en estrategias:
          def export_indicators(self):
              return self._export_indicators_helper(
                  self.bb.upperband, self.bb.middleband, self.bb.lowerband,
                  names=['BB_Upper','BB_Middle','BB_Lower']
              )

        Params:
          *indicators: indicadores/lÃ­neas de Backtrader a exportar.
          names: lista opcional de nombres (si no se proporciona, usa L1, L2, ...).

        Returns:
          dict con los indicadores exportados como listas.
        """
        indicators_dict = {}
        
        for i, ind in enumerate(indicators, 1):
            name = names[i-1] if names and i-1 < len(names) else f"L{i}"
            
            if hasattr(ind, 'array') and hasattr(ind, '__len__') and len(ind):
                indicators_dict[name] = list(ind.array)
                print(f"[EXPORT] {name}: {len(indicators_dict[name])} puntos")
            else:
                print(f"[EXPORT][SKIP] {name}: sin datos o no es lÃ­nea Backtrader")
        
        print(f"[EXPORT] Series: {list(indicators_dict.keys())}")
        return indicators_dict

    def _tag_order_for_parity(self, order, bar_index: int, ts_event: str, timeframe: str) -> None:
        """
        Guarda metadatos del envÃ­o para que el fill loguee el mismo bar_index
        que el bot (usa el bar del submit/signal en Ã³rdenes de mercado).
        """
        if order is None:
            return
        try:
            self._parity_order_meta[int(order.ref)] = {
                "bar_index": int(bar_index),
                "ts_event": str(ts_event),
                "timeframe": str(timeframe),
            }
        except Exception:
            pass

    # ---------------- Notificaciones: Ã³rdenes y trades ------------
    def notify_order(self, order):
        """
        Guarda una traza detallada de las Ã³rdenes COMPLETED para reconstruir trades.
        """
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            self._orders_log.append({
                'dt': order.data.datetime.datetime(0),
                'size': order.executed.size,
                'price': order.executed.price,
                'value': order.executed.value,
                'comm': order.executed.comm,
                'is_buy': order.isbuy(),
                'ref': order.ref,
                'data_name': getattr(order.data, '_name', 'data0'),
            })
            # Evento de fill
            if hasattr(self, "_event_logger"):
                symbol = getattr(order.data, "_symbol", getattr(order.data, "_name", "data0"))
                timeframe = getattr(order.data, "_timeframe_label", "entry")
                bar_idx = len(order.data) - 1 if hasattr(order.data, "__len__") else -1
                ts_event = order.data.datetime.datetime(0).isoformat()
                meta = None
                if hasattr(self, "_parity_order_meta"):
                    try:
                        meta = self._parity_order_meta.get(int(order.ref))
                    except Exception:
                        meta = None
                if meta:
                    bar_idx = int(meta.get("bar_index", bar_idx))
                    ts_event = str(meta.get("ts_event", ts_event))
                    timeframe = str(meta.get("timeframe", timeframe))
                side = "long" if order.isbuy() else "short"
                price = float(order.executed.price)
                # Paridad con bot: size siempre en magnitud positiva (side indica direcciÃ³n).
                size = abs(float(order.executed.size))
                self._event_logger.log({
                    "ts_event": ts_event,
                    "bar_index": bar_idx,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "event_type": "order_fill",
                    "side": side,
                    "price_ref": "close",
                    "price": price,
                    "size": size,
                    "commission": float(order.executed.comm or 0.0),
                    "slippage": 0.0,
                    "reason": "fill",
                    "order_id": str(order.ref),
                })
                # Estado de posiciÃ³n tras el fill
                try:
                    pos = self.getposition(order.data)
                    pos_size = float(pos.size)
                except Exception:
                    pos_size = 0.0
                pos_side = "long" if pos_size > 0 else "short" if pos_size < 0 else "flat"
                self._event_logger.log({
                    "ts_event": ts_event,
                    "bar_index": bar_idx,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "event_type": "position",
                    "side": pos_side,
                    "price_ref": "close",
                    "price": price,
                    "size": abs(pos_size),
                    "commission": float(order.executed.comm or 0.0),
                    "slippage": 0.0,
                    "reason": "position_update",
                    "order_id": str(order.ref),
                })
                if meta and hasattr(self, "_parity_order_meta"):
                    try:
                        self._parity_order_meta.pop(int(order.ref), None)
                    except Exception:
                        pass
            return

        # Canceled / Margin / Rejected -> no guardamos mÃ¡s info (pero podrÃ­as loguear si lo deseas)

    def notify_trade(self, trade):
        """
        Reconstruye y guarda un trade CERRADO basÃ¡ndose en el log de Ã³rdenes
        realizadas EN EL PERIODO del trade y sÃ³lo del feed correspondiente.
        """
        if not trade.isclosed:
            return

        open_dt = bt.num2date(trade.dtopen)
        close_dt = bt.num2date(trade.dtclose)
        data_name = getattr(trade.data, '_name', None)

        # Filtra por rango temporal y por activo
        related_orders = [
            o for o in self._orders_log
            if open_dt <= o['dt'] <= close_dt and (data_name is None or o.get('data_name') == data_name)
        ]

        if not related_orders:
            # Fallback: sin matching de Ã³rdenes, usamos datos del objeto trade (como ya hacÃ­as)
            self._closed_trades.append({
                'open_datetime': open_dt,
                'close_datetime': close_dt,
                'size': 0.0,
                'price_open': trade.price,
                'price_close': trade.data.close[0],
                'pnl': float(trade.pnl),
                'pnl_comm': float(trade.pnlcomm),
                'barlen': int(trade.barlen),
            })
            return

        # Agrupar por fecha (abre/cierra puede estar compuesto de varias Ã³rdenes)
        from collections import defaultdict
        by_dt = defaultdict(list)
        for o in related_orders:
            by_dt[o['dt']].append(o)

        dts = sorted(by_dt.keys())
        open_date = dts[0]
        close_date = dts[-1]

        # Precio medio ponderado en la apertura
        open_orders = by_dt[open_date]
        
        # Filtrar Ã³rdenes de apertura: cuando hay close() seguido de buy() en la misma barra,
        # necesitamos identificar solo las Ã³rdenes de apertura reales del trade
        # Calculamos el tamaÃ±o neto para determinar la direcciÃ³n del trade
        net_size = sum(o['size'] for o in open_orders)
        
        # Filtrar segÃºn la direcciÃ³n del trade:
        # - Si net_size > 0: trade largo, solo Ã³rdenes de compra (is_buy=True, size>0)
        # - Si net_size < 0: trade corto, solo Ã³rdenes de venta (is_buy=False, size<0)
        if net_size > 0:
            # Trade largo: solo Ã³rdenes de compra (excluye Ã³rdenes de cierre de venta)
            opening_orders = [o for o in open_orders if o['is_buy'] and o['size'] > 0]
        elif net_size < 0:
            # Trade corto: solo Ã³rdenes de venta (excluye Ã³rdenes de cierre de compra)
            opening_orders = [o for o in open_orders if not o['is_buy'] and o['size'] < 0]
        else:
            # Si net_size es 0, usar todas las Ã³rdenes (fallback)
            opening_orders = open_orders
        
        # Si no encontramos Ã³rdenes de apertura filtradas, usar todas (fallback)
        if not opening_orders:
            opening_orders = open_orders
        
        # Si hay mÃºltiples Ã³rdenes del mismo tipo en la misma fecha,
        # tomar solo la Ãºltima (la que realmente abriÃ³ el trade)
        # Esto ocurre cuando close() y buy()/sell() se ejecutan en la misma barra
        if len(opening_orders) > 1:
            signs = set(1 if o['size'] > 0 else -1 for o in opening_orders)
            if len(signs) == 1:
                # Todas son del mismo tipo: tomar solo la Ãºltima orden
                # (la Ãºltima en ejecutarse es la que abriÃ³ el trade)
                opening_orders = [opening_orders[-1]]
        
        # Calcular tamaÃ±o desde las Ã³rdenes de apertura filtradas
        size_open = sum(o['size'] for o in opening_orders)
        
        total_val_open = sum(abs(o['size']) * o['price'] for o in opening_orders)
        total_size_open = sum(abs(o['size']) for o in opening_orders)
        price_open = total_val_open / total_size_open if total_size_open > 0 else opening_orders[0]['price']

        # Precio de cierre: de la Ãºltima orden
        close_orders = by_dt[close_date]
        price_close = close_orders[-1]['price']

        self._closed_trades.append({
            'open_datetime': open_date,
            'close_datetime': close_date,
            'size': float(size_open),  # Usar tamaÃ±o calculado desde Ã³rdenes de apertura filtradas
            'price_open': float(price_open),
            'price_close': float(price_close),
            'pnl': float(trade.pnl),
            'pnl_comm': float(trade.pnlcomm),
            'barlen': int(trade.barlen),
        })

# ================================================================
# ESTRATEGIAS (lÃ³gica original conservada)
# ================================================================

'''
# ================================================================
# ESTRATEGIA DE PRUEBA: PivotZoneTest (ANTIGUA SOLO DETECTA ZONAS)
# ================================================================
# Constante fuera de la clase (requerido por Python para generator expressions)
_PIVOT_ZONE_MAX = 30

class PivotZoneTestIndicator(_BaseLoggedStrategy):
    """
    ============================
    ESTRATEGIA: TEST DE ZONAS PIVOTE
    ============================

    Evalua el funcionamiento de PivotZone.
    Guarda las zonas validadas en una lista interna (persistencia).
    Las zonas NUNCA se borran.
    Plotea lineas top/bot de cada zona guardada.

    NO EJECUTA ORDENES - Solo para visualizacion.
    """

    # Numero maximo de zonas a plotear (limitacion de Backtrader)
    MAX_ZONES = _PIVOT_ZONE_MAX

    # Lineas dinamicas: 2 por zona (top, bot)
    lines = tuple(
        f'zone_{i}_{side}' 
        for i in range(_PIVOT_ZONE_MAX) 
        for side in ('top', 'bot')
    )

    params = (
        ('n1', 14),       # Periodo ATR INTRADIA (2-10)
        ('n2', 60),       # Ancho zona % ATR
        ('n3', 3),        # Min pivotes para zona valida
        ('size_pct', 0.05),  # No usado, requerido por interface
    )

    plotinfo = dict(subplot=False)

    # 30 colores distintos para zonas (no se repiten hasta la zona 30)
    _zone_colors = [
        '#e6194b',  # rojo intenso
        '#3cb44b',  # verde
        '#4363d8',  # azul
        '#ffe119',  # amarillo
        '#f58231',  # naranja
        '#911eb4',  # morado
        '#42d4f4',  # cian
        '#f032e6',  # magenta
        '#bfef45',  # lima
        '#fabed4',  # rosa claro
        '#469990',  # verde azulado (teal)
        '#dcbeff',  # lavanda
        '#9a6324',  # marrÃ³n
        '#fffac8',  # crema
        '#800000',  # granate
        '#aaffc3',  # menta
        '#808000',  # oliva
        '#ffd8b1',  # melocotÃ³n
        '#000075',  # azul marino
        '#a9a9a9',  # gris
        '#b15928',  # Ã³xido
        '#1f78b4',  # azul medio
        '#33a02c',  # verde medio
        '#fb9a99',  # salmÃ³n
        '#e31a1c',  # rojo oscuro
        '#6a3d9a',  # violeta oscuro
        '#ff7f00',  # naranja fuerte
        '#b2df8a',  # verde pastel
    ]


    # Generar plotlines estaticamente a nivel de clase
    plotlines = {}
    for _i in range(_PIVOT_ZONE_MAX):
        _color = _zone_colors[_i % len(_zone_colors)]
        plotlines[f'zone_{_i}_top'] = dict(ls='--', color=_color, linewidth=1.0)
        plotlines[f'zone_{_i}_bot'] = dict(ls='--', color=_color, linewidth=1.0)

    def __init__(self):
        if not getattr(self, '_is_multi_ctx', False):
            super().__init__()

        # Indicador PivotZone existente
        self.pz = PivotZone(
            self.data,
            n1=self.p.n1,
            n2=self.p.n2,
            n3=self.p.n3,
        )

        # ============================================
        # PERSISTENCIA: Lista de zonas guardadas
        # Cada zona es un dict: {'top': float, 'bot': float, 'bar': int}
        # NUNCA se borran
        # ============================================
        self._saved_zones: List[Dict[str, float]] = []

        # Flag para detectar transicion valid: 0 -> 1
        self._prev_valid = 0.0

        print(f"[INIT] PivotZoneTest | n1={self.p.n1} n2={self.p.n2} n3={self.p.n3}")

    def next(self):
        """
        En cada barra:
        1. Detectar si PivotZone acaba de validar una zona (valid: 0->1)
        2. Si es asi, guardar la zona en _saved_zones (PERSISTENCIA)
        3. Publicar TODAS las zonas guardadas en las lineas de salida
        """
        current_valid = float(self.pz.valid[0])
        current_top = float(self.pz.top[0])
        current_bot = float(self.pz.bot[0])

        # ============================================
        # 1) Detectar nueva zona validada (transicion 0 -> 1)
        # ============================================
        if current_valid == 1.0 and self._prev_valid == 0.0:
            # Nueva zona validada - VERIFICAR si no se solapa con zonas existentes
            if math.isfinite(current_top) and math.isfinite(current_bot):
                new_mid = (current_top + current_bot) / 2.0
                new_width = current_top - current_bot
                min_distance = new_width * max(0.0, float(self.p.n1))
                
                # Verificar solapamiento con zonas guardadas
                is_valid_zone = True
                for existing_zone in self._saved_zones:
                    existing_mid = (existing_zone['top'] + existing_zone['bot']) / 2.0
                    distance = abs(new_mid - existing_mid)
                    
                    if distance < min_distance:
                        # Zona demasiado cercana - RECHAZAR
                        is_valid_zone = False
                        print(f"[ZONE REJECTED] Bar={len(self.data)} | "
                              f"Mid={new_mid:.6f} demasiado cerca de zona existente "
                              f"(dist={distance:.6f} < min={min_distance:.6f})")
                        break
                
                if is_valid_zone:
                    new_zone = {
                        'top': current_top,
                        'bot': current_bot,
                        'bar': len(self.data),
                    }
                    self._saved_zones.append(new_zone)
                    print(f"[ZONE SAVED] Bar={len(self.data)} | "
                          f"Top={current_top:.6f} Bot={current_bot:.6f} | "
                          f"Total zonas={len(self._saved_zones)}")
                
                # SIEMPRE resetear para buscar nueva zona (aunque se rechace)
                self.pz.reset()

        # Actualizar flag para siguiente barra
        self._prev_valid = current_valid

        # ============================================
        # 2) Publicar TODAS las zonas guardadas
        # ============================================
        for i in range(self.MAX_ZONES):
            top_line = getattr(self.lines, f'zone_{i}_top')
            bot_line = getattr(self.lines, f'zone_{i}_bot')

            if i < len(self._saved_zones):
                zone = self._saved_zones[i]
                top_line[0] = zone['top']
                bot_line[0] = zone['bot']
            else:
                top_line[0] = float('nan')
                bot_line[0] = float('nan')

    def get_saved_zones(self) -> List[Dict[str, float]]:
        """Retorna lista de todas las zonas guardadas (para uso externo)."""
        return self._saved_zones.copy()

    def export_indicators(self):
        """Exporta las lineas de zonas para plotting externo."""
        indicators = {}
        colors = {}  # Metadata de colores: mismo color para top y bot de cada zona

        for i in range(min(len(self._saved_zones), self.MAX_ZONES)):
            top_line = getattr(self.lines, f'zone_{i}_top')
            bot_line = getattr(self.lines, f'zone_{i}_bot')
            # Color ciclico para esta zona
            zone_color = self._zone_colors[i % len(self._zone_colors)]

            if hasattr(top_line, 'array') and len(top_line):
                indicators[f'Zone_{i}_Top'] = list(top_line.array)
                indicators[f'Zone_{i}_Bot'] = list(bot_line.array)
                # Asignar el mismo color a top y bot de la misma zona
                colors[f'Zone_{i}_Top'] = zone_color
                colors[f'Zone_{i}_Bot'] = zone_color

        # AÃ±adir metadata de colores
        indicators['_colors'] = colors

        print(f"[EXPORT] {len(self._saved_zones)} zonas guardadas, colores: {len(colors)}")
        return indicators
'''
# ================================================================
# ESTRATEGIA: PivotZoneTest (Zonas pivote: Ruptura + 2 cierres + trailing estructural)
# ================================================================
# Constante fuera de la clase (requerido por Python para generator expressions)
_PIVOT_ZONE_MAX = 30

class PivotZoneTest(_BaseLoggedStrategy):
    """
    ============================
    ESTRATEGIA: ZONAS PIVOTE (ZONE-TO-ZONE)
    ============================

    OBJETIVO
    --------
    Operar un recorrido â€œzone-to-zoneâ€ con lÃ³gica mÃ­nima y sin look-ahead:

      1) ZONAS (TF_zone)
         - Se construyen con PivotZone en TF_zone.
         - Se guardan en _saved_zones cuando PivotZone valida (valid 0->1).
         - Las zonas NO se borran.
         - En TF_entry siempre se usan los bordes vigentes de la Ãºltima zona guardada.

      2) ENTRADA (TF_entry)
         - Ruptura + confirmaciÃ³n de 2 cierres consecutivos fuera:
             * LONG:  close[-1] > Top y close[0] > Top
             * SHORT: close[-1] < Bot y close[0] < Bot

      3) STOP INICIAL (TF_stop)
         - LONG: Ãºltimo swing low confirmado en TF_stop dentro de la zona rota
         - SHORT: Ãºltimo swing high confirmado en TF_stop dentro de la zona rota

      4) TAKE PROFIT (zone-to-zone)
         - Zona objetivo: zona pivote mÃ¡s cercana â€œen direcciÃ³nâ€ (por nivel) ya existente.
         - TP: orden LÃMITE en el borde mÃ¡s cercano de la zona objetivo (por distancia al precio de entrada).
         - EjecuciÃ³n por â€œtoqueâ€ (por high/low, Backtrader lo simula con orden Limit).

      5) GESTIÃ“N DE SALIDA (SIN TRAILING)
         - Salida por stop inicial o TP de zona objetivo.

    MULTI-TIMEFRAME (IMPORTANTE)
    ----------------------------
    Para que la estrategia â€œcorraâ€ en TF_entry (next() por cada vela de entrada),
    lo habitual es que TF_entry sea data0 (self.data). AquÃ­ asumimos:

        self.TF_entry = self.datas[0]
        self.TF_zone  = self.datas[1]
        self.TF_stop  = self.datas[2]

    Si tu make_multi entrega otro orden, ajusta SOLO esas 3 asignaciones.
    """

    # Numero maximo de zonas a plotear (limitacion de Backtrader)
    MAX_ZONES = _PIVOT_ZONE_MAX

    # Lineas dinamicas: 2 por zona (top, bot)
    lines = tuple(
        f'zone_{i}_{side}'
        for i in range(_PIVOT_ZONE_MAX)
        for side in ('top', 'bot')
    )

    params = (
        ('n1', 3),          # Multiplicador de min_distance entre zonas (antes fijo en 3.0)
        ('n2', 60),         # Ancho zona % ATR
        ('n3', 3),          # Min pivotes para zona valida
        ('size_pct', 0.05), # Riesgo/porcentaje (usado por size_percent_by_stop)
        ('p', 0.50),        # Filtro â€œclose cerca del extremoâ€ (0.70 = bastante estricto)
    )

    plotinfo = dict(subplot=False)

    # 30 colores distintos para zonas (no se repiten hasta la zona 30)
    _zone_colors = [
        '#e6194b',  # rojo intenso
        '#3cb44b',  # verde
        '#4363d8',  # azul
        '#ffe119',  # amarillo
        '#f58231',  # naranja
        '#911eb4',  # morado
        '#42d4f4',  # cian
        '#f032e6',  # magenta
        '#bfef45',  # lima
        '#fabed4',  # rosa claro
        '#469990',  # verde azulado (teal)
        '#dcbeff',  # lavanda
        '#9a6324',  # marrÃ³n
        '#fffac8',  # crema
        '#800000',  # granate
        '#aaffc3',  # menta
        '#808000',  # oliva
        '#ffd8b1',  # melocotÃ³n
        '#000075',  # azul marino
        '#a9a9a9',  # gris
        '#b15928',  # Ã³xido
        '#1f78b4',  # azul medio
        '#33a02c',  # verde medio
        '#fb9a99',  # salmÃ³n
        '#e31a1c',  # rojo oscuro
        '#6a3d9a',  # violeta oscuro
        '#ff7f00',  # naranja fuerte
        '#b2df8a',  # verde pastel
    ]

    # Generar plotlines estaticamente a nivel de clase
    plotlines = {}
    for _i in range(_PIVOT_ZONE_MAX):
        _color = _zone_colors[_i % len(_zone_colors)]
        plotlines[f'zone_{_i}_top'] = dict(ls='--', color=_color, linewidth=1.0)
        plotlines[f'zone_{_i}_bot'] = dict(ls='--', color=_color, linewidth=1.0)

    # -------------------------
    # INIT
    # -------------------------
    def __init__(self):
        # CondiciÃ³n multi-context (misma convenciÃ³n que tus otras estrategias)
        if not getattr(self, '_is_multi_ctx', False):
            super().__init__()
        # Logger de eventos JSONL para paridad (no altera la logica de trading)

        events_path_env = (os.environ.get("BACKTEST_EVENTS_PATH") or "").strip()
        if events_path_env:
            events_path = Path(events_path_env)
            if not events_path.is_absolute():
                events_path = (Path(__file__).resolve().parent / events_path).resolve()
        else:
            events_path = Path(__file__).resolve().parent / "outputs" / "backtest_events.jsonl"
        self._event_logger = EventLogger(str(events_path))




        # ============================================================
        # 0) MAPEO MULTI-TF (ver docstring)
        # ============================================================
        self.TF_entry = self.datas[0]
        self.TF_zone  = self.datas[1] if len(self.datas) > 1 else self.datas[0]
        self.TF_stop  = self.datas[2] if len(self.datas) > 2 else self.datas[0]

        # ============================================================
        # 1) Indicador PivotZone (SE CALCULA EN TF_zone)
        #    Nota: Esto â€œcreaâ€ el indicador en init. El guardado/persistencia
        #    de zonas se hace cuando TF_zone avanza (no en cada vela de entrada).
        # ============================================================
        self.pz = PivotZone(
            self.TF_zone,
            n1=14,
            n2=self.p.n2,
            n3=self.p.n3,
        )

        # ============================================
        # PERSISTENCIA: Lista de zonas guardadas
        # Cada zona: {'top': float, 'bot': float, 'bar': int}
        # NUNCA se borran
        # ============================================
        self._saved_zones: List[Dict[str, float]] = []

        # Flag para detectar transiciÃ³n valid: 0 -> 1 (EN TF_zone)
        self._prev_valid = 0.0

        # â€œRelojesâ€ para no recalcular cosas en cada vela de TF_entry
        self._last_tf_zone_len = 0  # Ãºltimo len(TF_zone) procesado
        self._last_tf_stop_len = 0  # Ãºltimo len(TF_stop) procesado

        # ============================================================
        # 2) Swings para stop (SE CALCULAN EN TF_stop)
        #    Usamos Pivot3Candle (3 velas) sin mirar el futuro.
        #    Guardamos el Ãºltimo swing confirmado (high/low).
        # ============================================================
        self.pivot_stop = Pivot3Candle(self.TF_stop)
        self._stop_last_max: Optional[float] = None
        self._stop_last_min: Optional[float] = None
        self._stop_pivot_maxs: List[float] = []
        self._stop_pivot_mins: List[float] = []

        # ============================================================
        # 3) Estado de trading y Ã³rdenes vivas
        # ============================================================
        self._entry_order = None
        self._stop_order = None
        self._tp_order = None

        # DirecciÃ³n del trade actual: +1 long, -1 short, 0 none
        self._trade_dir: int = 0

        # Congelados al abrir trade (obligatorio por tu especificaciÃ³n)
        self._frozen_zone_broken: Optional[Dict[str, float]] = None
        self._frozen_zone_target: Optional[Dict[str, float]] = None

        # Stop activo actual (para â€œno retrocederâ€)
        self._active_stop_price: Optional[float] = None

        print(
            f"[INIT] PivotZoneTest | "
            f"TF_entry={getattr(self.TF_entry, '_name', 'data0')} "
            f"TF_zone={getattr(self.TF_zone,  '_name', 'data1')} "
            f"TF_stop={getattr(self.TF_stop,  '_name', 'data2')} | "
            f"min_distance_mult={self.p.n1} n2={self.p.n2} n3={self.p.n3} size={self.p.size_pct} p={self.p.p}"
        )

    def _emit_event(self, **kwargs):
        """Wrapper seguro para escribir eventos JSONL sin afectar el backtest."""
        try:
            self._event_logger.log(kwargs)
        except Exception:
            pass

    # ============================================================
    # HELPERS DE ZONAS
    # ============================================================
    def _get_current_zone(self) -> Optional[Dict[str, float]]:
        """
        Zona â€œvigenteâ€ = Ãºltima zona guardada (por tiempo).
        Si no hay zonas todavÃ­a, no se puede operar.
        """
        if not self._saved_zones:
            return None
        return self._saved_zones[-1]

    def _select_target_zone(self, broken_zone: Dict[str, float], direction: int, entry_ref: float) -> Optional[Dict[str, float]]:
        """
        Selecciona la zona objetivo ya EXISTENTE mÃ¡s cercana â€œen direcciÃ³nâ€ por nivel.

        - LONG: zona con MID por encima del MID de la zona rota y la mÃ¡s cercana.
        - SHORT: zona con MID por debajo del MID de la zona rota y la mÃ¡s cercana.

        NOTA:
        Esto evita â€œtarget a zona futuraâ€ (no conocida en el momento de entrada).
        """
        if not self._saved_zones or broken_zone is None:
            return None

        broken_mid = (float(broken_zone['top']) + float(broken_zone['bot'])) / 2.0

        best = None
        best_dist = None

        for z in self._saved_zones:
            if z is broken_zone:
                continue
            mid = (float(z['top']) + float(z['bot'])) / 2.0

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

    def _check_zone_breakout(self) -> Optional[tuple]:
        """
        Busca si el precio estÃ¡ rompiendo alguna zona guardada.
        
        VALIDACIÃ“N DE ORIGEN: Solo considera ruptura vÃ¡lida si el precio estaba 
        DENTRO de la zona al menos en 1 de las Ãºltimas 3-5 velas (evita micro-trades).
        
        Retorna:
            (zone, direction) si hay ruptura vÃ¡lida, None si no hay ruptura.
            - direction: +1 = LONG (ruptura hacia ARRIBA desde dentro)
                        -1 = SHORT (ruptura hacia ABAJO desde dentro)
        
        LÃ³gica:
            - Ruptura hacia ARRIBA: precio venÃ­a de dentro + 2 cierres por encima + follow-through
            - Ruptura hacia ABAJO: precio venÃ­a de dentro + 2 cierres por debajo + follow-through
            - Solo opera si existe una zona objetivo en la direcciÃ³n correspondiente
        """
        if not self._saved_zones:
            return None
        
        # Necesitamos al menos 5 velas de historial para validar origen
        if len(self.TF_entry) < 5:
            return None
        
        c0 = float(self.TF_entry.close[0])
        
        # Iterar sobre todas las zonas guardadas para buscar rupturas
        for zone in self._saved_zones:
            top = float(zone['top'])
            bot = float(zone['bot'])
            
            # ========================================
            # VALIDAR ORIGEN: Al menos 2 de las 3 Ãºltimas velas (-1, -2, -3)
            # deben estar dentro de la zona (evita micro-trades)
            # ========================================
            if len(self.TF_entry) >= 4:  # Necesitamos al menos velas hasta -3
                count_inside = sum(
                    1 for i in [1, 2, 3]
                    if bot <= float(self.TF_entry.close[-i]) <= top
                )
                was_inside = count_inside >= 2  # Al menos 2 de 3
            else:
                was_inside = False
            
            # Si el precio no vino desde dentro de la zona, ignorar esta zona
            if not was_inside:
                continue
            
            # ========================================
            # RUPTURA HACIA ARRIBA â†’ LONG
            # ========================================
            # Condiciones:
            # 0) El precio venÃ­a de DENTRO de la zona (ya validado arriba)
            # 1) Vela [-1]: Ruptura limpia (cierre cerca del high)
            # 2) Vela [0]: ConfirmaciÃ³n (se mantiene fuera)
            # 3) Follow-through: precio continÃºa en direcciÃ³n (con tolerancia)
            long_ok = (
                # ConfirmaciÃ³n de ruptura: 2 cierres consecutivos fuera por arriba.
                (float(self.TF_entry.close[-1]) > top) and
                (c0 > top)
            )
            
            if long_ok:
                # Verificar que existe una zona objetivo mÃ¡s arriba
                target = self._select_target_zone(zone, direction=+1, entry_ref=c0)
                if target is not None:
                    return (zone, +1)
            
            # ========================================
            # RUPTURA HACIA ABAJO â†’ SHORT
            # ========================================
            # Condiciones:
            # 0) El precio venÃ­a de DENTRO de la zona (ya validado arriba)
            # 1) Vela [-1]: Ruptura limpia (cierre cerca del low)
            # 2) Vela [0]: ConfirmaciÃ³n (se mantiene fuera)
            # 3) Follow-through: precio continÃºa en direcciÃ³n (con tolerancia)
            short_ok = (
                # ConfirmaciÃ³n de ruptura: 2 cierres consecutivos fuera por abajo.
                (float(self.TF_entry.close[-1]) < bot) and
                (c0 < bot)
            )
            
            if short_ok:
                # Verificar que existe una zona objetivo mÃ¡s abajo
                target = self._select_target_zone(zone, direction=-1, entry_ref=c0)
                if target is not None:
                    return (zone, -1)
        
        return None

    def _tp_edge_nearest(self, target_zone: Dict[str, float], entry_price: float) -> float:
        """
        Borde â€œmÃ¡s cercanoâ€ de la zona objetivo por distancia al precio de entrada.
        """
        top = float(target_zone['top'])
        bot = float(target_zone['bot'])
        d_top = abs(top - float(entry_price))
        d_bot = abs(bot - float(entry_price))
        return bot if d_bot <= d_top else top

    def _find_stop_inside_broken_zone(self, broken_zone: Dict[str, float], direction: int) -> Optional[float]:
        """Busca el Ãºltimo pivote de TF_stop que cae dentro de la zona rota."""
        top = float(broken_zone["top"])
        bot = float(broken_zone["bot"])
        pivots = self._stop_pivot_mins if direction > 0 else self._stop_pivot_maxs
        for price in reversed(pivots):
            p = float(price)
            if bot <= p <= top:
                return p
        return None

    # ============================================================
    # HELPERS DE CONFIRMACIÃ“N (TF_entry)
    # ============================================================
    def _close_pos(self, idx: int = 0) -> float:
        """
        pos = (Close - Low) / (High - Low)
        Si high==low (rango 0), devolvemos 0.5 por neutralidad.
        """
        h = float(self.TF_entry.high[idx])
        l = float(self.TF_entry.low[idx])
        c = float(self.TF_entry.close[idx])

        rng = h - l
        if rng <= 0.0:
            return 0.5
        return (c - l) / rng

    def _outside_long(self, top: float, idx: int = 0) -> bool:
        """
        Para LONG: cierre fuera + close cerca del high (pos > p).
        """
        c = float(self.TF_entry.close[idx])
        return (c > float(top)) and (self._close_pos(idx) > float(self.p.p))

    def _outside_short(self, bot: float, idx: int = 0) -> bool:
        """
        Para SHORT: cierre fuera + close cerca del low (pos < 1-p).
        """
        c = float(self.TF_entry.close[idx])
        return (c < float(bot)) and (self._close_pos(idx) < (1.0 - float(self.p.p)))

    # ============================================================
    # HELPERS DE Ã“RDENES / RESET
    # ============================================================
    def _cancel_if_alive(self, o):
        """
        CancelaciÃ³n defensiva: Backtrader ignora cancel si no estÃ¡ viva,
        pero evitamos ruido y estados raros.
        """
        if o is None:
            return
        try:
            if o.alive():
                self.cancel(o)
        except Exception:
            # No reventar el backtest por un cancel defensivo
            pass

    def _reset_trade_state(self):
        """
        Limpia todo el estado del trade al cerrarse (por TP o por STOP).
        """
        self._entry_order = None
        self._stop_order = None
        self._tp_order = None
        self._trade_dir = 0
        self._frozen_zone_broken = None
        self._frozen_zone_target = None
        self._active_stop_price = None

    def _replace_stop(self, new_stop_price: float):
        """
        Reemplaza el stop existente por uno nuevo (trailing estructural).
        Regla: el stop nunca retrocede.
        """
        if self.position.size == 0:
            return

        size = abs(self.position.size)

        # Cancelar stop anterior si sigue vivo
        self._cancel_if_alive(self._stop_order)

        # Crear nuevo stop
        if self._trade_dir > 0:
            # LONG -> stop es una SELL Stop
            self._stop_order = self.sell(
                data=self.TF_entry,
                exectype=bt.Order.Stop,
                price=float(new_stop_price),
                size=size
            )
        elif self._trade_dir < 0:
            # SHORT -> stop es una BUY Stop
            self._stop_order = self.buy(
                data=self.TF_entry,
                exectype=bt.Order.Stop,
                price=float(new_stop_price),
                size=size
            )

        self._active_stop_price = float(new_stop_price)
        dt = self.TF_entry.datetime.datetime(0)
        print(f"[TRAIL] {dt} | Nuevo STOP={self._active_stop_price:.6f} | dir={self._trade_dir}")

    # ============================================================
    # NOTIFY_ORDER (colocar TP/STOP tras entrada y limpiar en salida)
    # ============================================================
    def notify_order(self, order):
        # Mantener (si existe) el logger/captura de la clase base
        try:
            super().notify_order(order)
        except Exception:
            pass

        if order.status in [order.Submitted, order.Accepted]:
            return

        dt = self.TF_entry.datetime.datetime(0)

        # -------------------------
        # 1) Entrada ejecutada
        # -------------------------
        if self._entry_order is not None and order.ref == self._entry_order.ref:
            if order.status == order.Completed:
                entry_price = float(order.executed.price)
                size = abs(float(order.executed.size))

                # Al ejecutarse la entrada, colocamos:
                # - Stop inicial (ya calculado en next usando TF_stop)
                # - TP limit hacia zona objetivo (borde mÃ¡s cercano)
                if self._frozen_zone_target is None or self._active_stop_price is None:
                    # Estado inconsistente (no deberÃ­a ocurrir); cerramos por seguridad.
                    print(f"[WARN] {dt} | Entrada ejecutada pero falta target/stop. Cerrando.")
                    self.close(data=self.TF_entry)
                    self._reset_trade_state()
                    return

                tp_price = float(self._tp_edge_nearest(self._frozen_zone_target, entry_price))

                # Stop (ya estÃ¡ en _active_stop_price)
                stop_price = float(self._active_stop_price)

                if self._trade_dir > 0:
                    # LONG: Stop=Sell Stop, TP=Sell Limit
                    self._stop_order = self.sell(
                        data=self.TF_entry,
                        exectype=bt.Order.Stop,
                        price=stop_price,
                        size=size
                    )
                    self._tp_order = self.sell(
                        data=self.TF_entry,
                        exectype=bt.Order.Limit,
                        price=tp_price,
                        size=size
                    )
                else:
                    # SHORT: Stop=Buy Stop, TP=Buy Limit
                    self._stop_order = self.buy(
                        data=self.TF_entry,
                        exectype=bt.Order.Stop,
                        price=stop_price,
                        size=size
                    )
                    self._tp_order = self.buy(
                        data=self.TF_entry,
                        exectype=bt.Order.Limit,
                        price=tp_price,
                        size=size
                    )

                print(
                    f"[ENTRY FILLED] {dt} | dir={self._trade_dir} | "
                    f"entry={entry_price:.6f} | stop={stop_price:.6f} | tp={tp_price:.6f}"
                )
            else:
                # Rechazada/cancelada
                print(f"[ENTRY FAIL] {dt} | status={order.getstatusname()}")
                self._reset_trade_state()
            return

        # -------------------------
        # 2) Stop ejecutado
        # -------------------------
        if self._stop_order is not None and order.ref == self._stop_order.ref:
            if order.status == order.Completed:
                exit_price = float(order.executed.price)
                print(f"[EXIT STOP] {dt} | price={exit_price:.6f}")

                # Cancelar TP si sigue vivo
                self._cancel_if_alive(self._tp_order)
                self._reset_trade_state()
            return

        # -------------------------
        # 3) TP ejecutado
        # -------------------------
        if self._tp_order is not None and order.ref == self._tp_order.ref:
            if order.status == order.Completed:
                exit_price = float(order.executed.price)
                print(f"[EXIT TP] {dt} | price={exit_price:.6f}")

                # Cancelar STOP si sigue vivo
                self._cancel_if_alive(self._stop_order)
                self._reset_trade_state()
            return

    # ============================================================
    # NEXT
    # ============================================================
    def next(self):
        """
        FLUJO (por cada vela de TF_entry):

        A) Si TF_zone avanzÃ³:
           - Leer pz.valid/top/bot (TF_zone)
           - Detectar 0->1, guardar zona, aplicar solapamiento, reset PivotZone

        B) Si TF_stop avanzÃ³:
           - Actualizar swings (max/min) con max_min_search en TF_stop
           - Si hay trade abierto: actualizar trailing stop SOLO con swings fuera de zona rota

        C) Publicar lÃ­neas de todas las zonas guardadas (plot)

        D) Si NO hay posiciÃ³n:
           - Evaluar ruptura + 2 cierres fuera + filtros (en TF_entry)
           - Congelar zona rota + zona objetivo
           - Calcular stop inicial con swings TF_stop
           - Calcular size con size_percent_by_stop
           - Enviar orden de entrada
        """

        # ============================================================
        # A) PROCESAR TF_zone SOLO CUANDO AVANZA (evita recalcular cada vela)
        # ============================================================
        tf_zone_len = len(self.TF_zone)
        if tf_zone_len != self._last_tf_zone_len:
            self._last_tf_zone_len = tf_zone_len

            current_valid = float(self.pz.valid[0])
            current_top = float(self.pz.top[0])
            current_bot = float(self.pz.bot[0])
            if os.getenv("PARITY_ZONE_STATE", "0") == "1":
                try:
                    max_bars = int(os.getenv("PARITY_ZONE_STATE_MAX", "0") or 0)
                except Exception:
                    max_bars = 0
                if max_bars <= 0 or tf_zone_len <= max_bars:
                    try:
                        locked = bool(getattr(self.pz, "_locked", False))
                        mid_val = getattr(self.pz, "_locked_mid", float("nan")) if locked else getattr(self.pz, "_pz_mid", float("nan"))
                        # SemÃ¡ntica estÃ¡ndar TF_zone (paridad):
                        # - width disponible incluso si mid/top/bot aÃºn no existen (fallback).
                        # - si la zona estÃ¡ bloqueada, se usa el ancho bloqueado.
                        if locked:
                            width_val = float(getattr(self.pz, "_locked_width", float("nan")))
                        else:
                            try:
                                width_val = float(self.pz._calc_width())
                            except Exception:
                                width_val = float("nan")
                        has_above = bool(getattr(self.pz, "_has_been_above", False))
                        has_below = bool(getattr(self.pz, "_has_been_below", False))
                        pivots_count = len(getattr(self.pz, "_pz_prices", []))
                        tf_label = getattr(self.TF_zone, "_timeframe_label", "zone")
                        ts_str = self.TF_zone.datetime.datetime(0).isoformat()
                        reason = (
                            f"valid={int(locked)} "
                            f"above={int(has_above)} "
                            f"below={int(has_below)} "
                            f"pivots={pivots_count}"
                        )
                        self._emit_event(
                            ts_event=ts_str,
                            bar_index=int(tf_zone_len - 1),
                            symbol=getattr(self.TF_entry, "_symbol", getattr(self.TF_entry, "_name", "data0")),
                            timeframe=tf_label,
                            event_type="zone_state",
                            side="flat",
                            price_ref="mid",
                            price=float(mid_val) if math.isfinite(float(mid_val)) else None,
                            size=float(width_val) if math.isfinite(float(width_val)) else None,
                            state_valid=int(locked),
                            state_has_been_above=int(has_above),
                            state_has_been_below=int(has_below),
                            state_pivots=int(pivots_count),
                            state_mid=float(mid_val) if math.isfinite(float(mid_val)) else None,
                            state_width=float(width_val) if math.isfinite(float(width_val)) else None,
                            commission=0.0,
                            slippage=0.0,
                            reason=reason,
                            order_id="",
                        )
                    except Exception:
                        pass
            try:
                # InstrumentaciÃ³n de paridad: pivotes confirmados en TF_zone (misma fuente que PivotZone usa internamente).
                pivot_price = getattr(self.pz, "_last_pivot_price", None)
                pivot_reason = getattr(self.pz, "_last_pivot_reason", "")
                if pivot_price is not None and math.isfinite(float(pivot_price)):
                    tf_label = getattr(self.TF_zone, "_timeframe_label", "zone")
                    ts_str = self.TF_zone.datetime.datetime(0).isoformat()
                    self._emit_event(
                        ts_event=ts_str,
                        bar_index=int(tf_zone_len - 1),
                        symbol=getattr(self.TF_entry, "_symbol", getattr(self.TF_entry, "_name", "data0")),
                        timeframe=tf_label,
                        event_type="pivot_confirmed",
                        side="flat",
                        price_ref="pivot",
                        price=float(pivot_price),
                        size=0.0,
                        commission=0.0,
                        slippage=0.0,
                        reason=pivot_reason,
                        order_id="",
                    )
            except Exception:
                pass

            # ============================================
            # 1) Detectar nueva zona validada (transicion 0 -> 1)
            # ============================================
            if current_valid == 1.0 and self._prev_valid == 0.0:
                # Nueva zona validada - VERIFICAR si no se solapa con zonas existentes
                if math.isfinite(current_top) and math.isfinite(current_bot):
                    new_mid = (current_top + current_bot) / 2.0
                    new_width = current_top - current_bot
                    min_distance_mult = max(0.0, float(self.p.n1))
                    min_distance = new_width * min_distance_mult

                    # Verificar solapamiento con zonas guardadas
                    is_valid_zone = True
                    for existing_zone in self._saved_zones:
                        existing_mid = (existing_zone['top'] + existing_zone['bot']) / 2.0
                        distance = abs(new_mid - existing_mid)

                        if distance < min_distance:
                            # Zona demasiado cercana - RECHAZAR
                            is_valid_zone = False
                            print(f"[ZONE REJECTED] Bar={tf_zone_len} | "
                                  f"Mid={new_mid:.6f} demasiado cerca de zona existente "
                                  f"(dist={distance:.6f} < min={min_distance:.6f})")
                            break

                    if is_valid_zone:
                        new_zone = {
                            'top': current_top,
                            'bot': current_bot,
                            'bar': tf_zone_len,
                        }
                        self._saved_zones.append(new_zone)
                        try:
                            tf_label = getattr(self.TF_zone, "_timeframe_label", "zone")
                            ts_str = self.TF_zone.datetime.datetime(0).isoformat()
                            self._emit_event(
                                ts_event=ts_str,
                                bar_index=int(tf_zone_len - 1),
                                symbol=getattr(self.TF_entry, "_symbol", getattr(self.TF_entry, "_name", "data0")),
                                timeframe=tf_label,
                                event_type="zone_saved",
                                side="flat",
                                price_ref="mid",
                                price=float(new_mid),
                                size=float(new_width),
                                commission=0.0,
                                slippage=0.0,
                                reason="zone_saved",
                                order_id="",
                            )
                        except Exception:
                            pass
                        print(f"[ZONE SAVED] Bar={tf_zone_len} | "
                              f"Top={current_top:.6f} Bot={current_bot:.6f} | "
                              f"Total zonas={len(self._saved_zones)}")

                    # SIEMPRE resetear para buscar nueva zona (aunque se rechace)
                    self.pz.reset()

            # Actualizar flag para siguiente vela de TF_zone
            self._prev_valid = current_valid

        # ============================================================
        # B) PROCESAR TF_stop SOLO CUANDO AVANZA (swings + trailing)
        # ============================================================
        tf_stop_len = len(self.TF_stop)
        if tf_stop_len != self._last_tf_stop_len:
            self._last_tf_stop_len = tf_stop_len

            # Obtener swings del indicador Pivot3Candle
            pivot_max = float(self.pivot_stop.lines.pivot_max[0])
            pivot_min = float(self.pivot_stop.lines.pivot_min[0])

            # Actualizar si hay nuevo swing confirmado (pivot_max/min son NaN cuando no hay pivote)
            if not math.isnan(pivot_max):
                self._stop_last_max = pivot_max
                self._stop_pivot_maxs.append(float(pivot_max))
                # print(f"[TF_STOP] Nuevo swing HIGH confirmado: {self._stop_last_max:.6f}")

            if not math.isnan(pivot_min):
                self._stop_last_min = pivot_min
                self._stop_pivot_mins.append(float(pivot_min))
                # print(f"[TF_STOP] Nuevo swing LOW confirmado: {self._stop_last_min:.6f}")

        # ============================================================
        # C) PUBLICAR TODAS LAS ZONAS GUARDADAS (PLOTEO)
        # ============================================================
        for i in range(self.MAX_ZONES):
            top_line = getattr(self.lines, f'zone_{i}_top')
            bot_line = getattr(self.lines, f'zone_{i}_bot')

            if i < len(self._saved_zones):
                zone = self._saved_zones[i]
                top_line[0] = zone['top']
                bot_line[0] = zone['bot']
            else:
                top_line[0] = float('nan')
                bot_line[0] = float('nan')

        # ============================================================
        # D) ENTRADAS (TF_entry) â€” solo si estamos flat y sin orden pendiente
        # ============================================================
        if self.position.size != 0:
            return
        if self._entry_order is not None:
            # Ya hay una entrada enviada y pendiente
            return
        if len(self.TF_entry) < 2:
            return

        # Buscar ruptura de CUALQUIER zona guardada
        breakout = self._check_zone_breakout()
        if breakout is None:
            return

        # Extraer la zona rota y la direcciÃ³n
        broken_zone, direction = breakout
        
        top = float(broken_zone['top'])
        bot = float(broken_zone['bot'])
        c0 = float(self.TF_entry.close[0])

        # Obtener zona objetivo (ya validada en _check_zone_breakout)
        target = self._select_target_zone(broken_zone, direction=direction, entry_ref=c0)
        if target is None:
            return

        # ------------------------------------------------------------
        # LONG: Ruptura hacia ARRIBA
        # ------------------------------------------------------------
        if direction > 0:
            # Stop inicial estructural = Ãºltimo swing low en TF_stop
            stop_price = self._find_stop_inside_broken_zone(broken_zone, direction=+1)
            if stop_price is None:
                return
            stop_price = float(stop_price)

            # Size por distancia a stop
            size = float(self.size_percent_by_stop(self.p.size_pct, stop_price, data=self.TF_entry))
            if size <= 0:
                return

            # Congelar zonas
            self._trade_dir = +1
            self._frozen_zone_broken = broken_zone
            self._frozen_zone_target = target
            self._active_stop_price = stop_price

            dt = self.TF_entry.datetime.datetime(0)
            ts_str = dt.isoformat()
            bar_idx = len(self.TF_entry) - 1
            symbol = getattr(self.TF_entry, '_symbol', getattr(self.TF_entry, '_name', 'data0'))
            tf_label = getattr(self.TF_entry, "_timeframe_label", "entry")
            self._emit_event(
                ts_event=ts_str,
                bar_index=bar_idx,
                symbol=symbol,
                timeframe=tf_label,
                event_type='signal',
                side='long',
                price_ref='close',
                price=c0,
                size=size,
                commission=0.0,
                slippage=0.0,
                reason='breakout_long',
                order_id=''
            )
            print(
                f"[SIGNAL LONG] {dt} | close={c0:.6f} > Top={top:.6f} | "
                f"Stop0={stop_price:.6f} | TargetTop={float(target['top']):.6f} TargetBot={float(target['bot']):.6f} | "
                f"size={size:.6f}"
            )

            self._entry_order = self.buy(data=self.TF_entry, size=size)
            self._tag_order_for_parity(self._entry_order, bar_idx, ts_str, tf_label)
            self._emit_event(
                ts_event=ts_str,
                bar_index=bar_idx,
                symbol=symbol,
                timeframe=tf_label,
                event_type='order_submit',
                side='long',
                price_ref='market',
                price=c0,
                size=size,
                commission=0.0,
                slippage=0.0,
                reason='submit_long',
                order_id=str(getattr(self._entry_order, 'ref', ''))
            )
            return

        # ------------------------------------------------------------
        # SHORT: Ruptura hacia ABAJO
        # ------------------------------------------------------------
        if direction < 0:
            # Stop inicial estructural = Ãºltimo swing high en TF_stop
            stop_price = self._find_stop_inside_broken_zone(broken_zone, direction=-1)
            if stop_price is None:
                return
            stop_price = float(stop_price)

            # Size por distancia a stop
            size = float(self.size_percent_by_stop(self.p.size_pct, stop_price, data=self.TF_entry))
            if size <= 0:
                return

            # Congelar zonas
            self._trade_dir = -1
            self._frozen_zone_broken = broken_zone
            self._frozen_zone_target = target
            self._active_stop_price = stop_price

            dt = self.TF_entry.datetime.datetime(0)
            ts_str = dt.isoformat()
            bar_idx = len(self.TF_entry) - 1
            symbol = getattr(self.TF_entry, '_symbol', getattr(self.TF_entry, '_name', 'data0'))
            tf_label = getattr(self.TF_entry, "_timeframe_label", "entry")
            self._emit_event(
                ts_event=ts_str,
                bar_index=bar_idx,
                symbol=symbol,
                timeframe=tf_label,
                event_type='signal',
                side='short',
                price_ref='close',
                price=c0,
                size=size,
                commission=0.0,
                slippage=0.0,
                reason='breakout_short',
                order_id=''
            )
            print(
                f"[SIGNAL SHORT] {dt} | close={c0:.6f} < Bot={bot:.6f} | "
                f"Stop0={stop_price:.6f} | TargetTop={float(target['top']):.6f} TargetBot={float(target['bot']):.6f} | "
                f"size={size:.6f}"
            )

            self._entry_order = self.sell(data=self.TF_entry, size=size)
            self._tag_order_for_parity(self._entry_order, bar_idx, ts_str, tf_label)
            self._emit_event(
                ts_event=ts_str,
                bar_index=bar_idx,
                symbol=symbol,
                timeframe=tf_label,
                event_type='order_submit',
                side='short',
                price_ref='market',
                price=c0,
                size=size,
                commission=0.0,
                slippage=0.0,
                reason='submit_short',
                order_id=str(getattr(self._entry_order, 'ref', ''))
            )
            return

    # ============================================================
    # EXPORTS (mantengo tu API)
    # ============================================================
    def get_saved_zones(self) -> List[Dict[str, float]]:
        """Retorna lista de todas las zonas guardadas (para uso externo)."""
        return self._saved_zones.copy()

    def export_indicators(self):
        """Exporta las lineas de zonas para plotting externo."""
        indicators = {}
        colors = {}  # Metadata de colores: mismo color para top y bot de cada zona

        for i in range(min(len(self._saved_zones), self.MAX_ZONES)):
            top_line = getattr(self.lines, f'zone_{i}_top')
            bot_line = getattr(self.lines, f'zone_{i}_bot')
            # Color ciclico para esta zona
            zone_color = self._zone_colors[i % len(self._zone_colors)]

            if hasattr(top_line, 'array') and len(top_line):
                indicators[f'Zone_{i}_Top'] = list(top_line.array)
                indicators[f'Zone_{i}_Bot'] = list(bot_line.array)
                # Asignar el mismo color a top y bot de la misma zona
                colors[f'Zone_{i}_Top'] = zone_color
                colors[f'Zone_{i}_Bot'] = zone_color

        # AÃ±adir metadata de colores
        indicators['_colors'] = colors

        print(f"[EXPORT] {len(self._saved_zones)} zonas guardadas, colores: {len(colors)}")
        return indicators


class DosMedias(_BaseLoggedStrategy):
    """
    Estrategia â€œsiempre en mercadoâ€ basada en cruce de dos SMAs.
      - n1: SMA rÃ¡pida
      - n2: SMA lenta
      - n3: no usado
      - size_pct: % de equity por operaciÃ³n 
    """
    params = (('n1', 7), ('n2', 20), ('n3', None), ('size_pct', 0.05))

    def __init__(self):
        # AÃ±adir esta condiciÃ³n en todas las estrategias que usen make_multi.
        if not getattr(self, '_is_multi_ctx', False):
            super().__init__()
        self.sma_fast = bt.talib.SMA(self.data.close, timeperiod=self.p.n1)
        self.sma_slow = bt.talib.SMA(self.data.close, timeperiod=self.p.n2)
        self.xover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        # Cruce alcista: media rÃ¡pida cruza por encima â†’ Largo
        if self.xover[0] > 0:
            self.close()
            self.buy(size=self.size_percent(self.p.size_pct))
        
        # Cruce bajista: media rÃ¡pida cruza por debajo â†’ Corto
        elif self.xover[0] < 0:
            self.close()
            self.sell(size=self.size_percent(self.p.size_pct))

    def export_indicators(self):
        return self._export_indicators_helper(
            self.sma_fast, self.sma_slow,
            names=['SMA_Fast', 'SMA_Slow']
        )

########################################################################################################################
class Tres_Medias_Filtered(_BaseLoggedStrategy):
    """
    Tendencial con filtro de pendiente de SMA3 y ADX>20:
      - Si pendiente(sma3) > +slope_th y ADX>20 -> sÃ³lo largos (al cruce sma1>sma2)
      - Si pendiente(sma3) < -slope_th y ADX>20 -> sÃ³lo cortos (al cruce sma1<sma2)
    """
    params = (('n1', 8), ('n2', 40), ('n3', 200), ('size_pct', 0.05), ('adx_period', 14), ('slope_th', 0.02))

    def __init__(self):
        # AÃ±adir esta condiciÃ³n en todas las estrategias que usen make_multi.
        if not getattr(self, '_is_multi_ctx', False):
            super().__init__()
        self.sma1 = bt.talib.SMA(self.data.close, timeperiod=self.p.n1)
        self.sma2 = bt.talib.SMA(self.data.close, timeperiod=self.p.n2)
        self.sma3 = bt.talib.SMA(self.data.close, timeperiod=self.p.n3)
        self.adx = bt.talib.ADX(self.data.high, self.data.low, self.data.close, timeperiod=self.p.adx_period)
        self.x12 = bt.indicators.CrossOver(self.sma1, self.sma2)

    def next(self):
        date = self.data.datetime.datetime(0)
        price = float(self.data.close[0])
        sma1v = float(self.sma1[0])
        sma2v = float(self.sma2[0])
        sma3v = float(self.sma3[0])
        adxv = float(self.adx[0])

        if len(self.sma3) < 2 or self.sma3[-1] == 0:
            return
        slope_pct = (self.sma3[0] - self.sma3[-1]) / self.sma3[-1] * 100.0

        # Largos
        if slope_pct > self.p.slope_th and adxv > 20 and self.x12[0] > 0:
            self.buy(size=self.size_percent(self.p.size_pct))
        # Cierre de largos
        if self.x12[0] < 0 and self.position.size > 0:
            self.close()

        # Cortos
        if slope_pct < -self.p.slope_th and adxv > 20 and self.x12[0] < 0:
            self.sell(size=self.size_percent(self.p.size_pct))
        # Cierre de cortos
        if self.x12[0] > 0 and self.position.size < 0:
            self.close()


    def export_indicators(self):
        return self._export_indicators_helper(
            self.sma1, self.sma2, self.sma3, self.adx,
            names=['SMA1', 'SMA2', 'SMA3', 'ADX']
        )

########################################################################################################################
class SMA_sobre_RSI(_BaseLoggedStrategy):
    """
    RSI con media mÃ³vil sobre RSI:
      - CORTO: RSI > n1 y RSI cruza ABAJO su SMA(n3)
      - LARGO: RSI < n2 y RSI cruza ARRIBA su SMA(n3)
      - Salidas: cruce contrario de RSI con su SMA.
    """
    params = (('n1', 60), ('n2', 40), ('n3', 14), ('size_pct', 0.05), ('rsi_period', 14))

    def __init__(self):
        # AÃ±adir esta condiciÃ³n en todas las estrategias que usen make_multi.
        if not getattr(self, '_is_multi_ctx', False):
            super().__init__()
        self.rsi = bt.talib.RSI(self.data.close, timeperiod=self.p.rsi_period)
        self.rsi_ma = bt.talib.SMA(self.rsi, timeperiod=self.p.n3)
        self.x_rsi_ma = bt.indicators.CrossOver(self.rsi, self.rsi_ma)
        #print(f"[INIT] SMA_sobre_RSI | n1={self.p.n1} n2={self.p.n2} n3={self.p.n3}")

    def next(self):
        date = self.data.datetime.datetime(0)
        price = float(self.data.close[0])
        rsiv = float(self.rsi[0])
        rsimv = float(self.rsi_ma[0])
        #print(f"[BAR ] {date} | Px={price:.6f} | RSI={rsiv:.2f} | RSI_MA={rsimv:.2f} | x={self.x_rsi_ma[0]}")

        # Entradas y salidas (idÃ©ntico a tu lÃ³gica)
        if rsiv > self.p.n1 and self.x_rsi_ma[0] < 0:
            #print("[SELL] RSI>n1 & RSI cruza abajo SMA -> CORTO")
            self.sell(size=self.size_percent(self.p.size_pct))
        if self.position.size < 0 and self.x_rsi_ma[0] > 0:
            #print("[CLOSE] TP corto (RSI cruza arriba SMA)")
            self.close()

        if rsiv < self.p.n2 and self.x_rsi_ma[0] > 0:
            #print("[BUY ] RSI<n2 & RSI cruza arriba SMA -> LARGO")
            self.buy(size=self.size_percent(self.p.size_pct))
        if self.position.size > 0 and self.x_rsi_ma[0] < 0:
            #print("[CLOSE] TP largo (RSI cruza abajo SMA)")
            self.close()


    def export_indicators(self):
        return self._export_indicators_helper(
            self.rsi, self.rsi_ma,
            names=['RSI', 'RSI_SMA']
        )

########################################################################################################################
class RSI_WMA_TurnFilters(_BaseLoggedStrategy):
    """
    RSI + giros de WMA con filtros:
      - BUY filter si RSI<n2 -> abre LARGO al giro de WMA hacia arriba.
      - SELL filter si RSI>n1 -> abre CORTO al giro de WMA hacia abajo.
      - Stops: mÃ­nimos/mÃ¡ximos previos (lookback=n3).
    """
    params = (('n1', 60), ('n2', 40), ('n3', 20), ('size_pct', 0.05), ('rsi_period', 14))

    def __init__(self):
        # AÃ±adir esta condiciÃ³n en todas las estrategias que usen make_multi.
        if not getattr(self, '_is_multi_ctx', False):
            super().__init__()
        self.rsi = bt.talib.RSI(self.data.close, timeperiod=self.p.rsi_period)
        self.wma = bt.talib.WMA(self.data.close, timeperiod=self.p.n3)
        self.piv = Pivot3Candle(self.data)
        self.buy_filter = False
        self.sell_filter = False
        self.stop_long = None
        self.stop_short = None

    def _wma_turns_up(self) -> bool:
        if len(self.wma) < 3:
            return False
        slope_prev = float(self.wma[-1]) - float(self.wma[-2])
        slope_now = float(self.wma[0]) - float(self.wma[-1])
        return (slope_prev < 0) and (slope_now > 0)

    def _wma_turns_down(self) -> bool:
        if len(self.wma) < 3:
            return False
        slope_prev = float(self.wma[-1]) - float(self.wma[-2])
        slope_now = float(self.wma[0]) - float(self.wma[-1])
        return (slope_prev > 0) and (slope_now < 0)

    def next(self):
        date = self.data.datetime.datetime(0)
        price = float(self.data.close[0])
        rsiv = float(self.rsi[0])
        wmav = float(self.wma[0])
        #print(f"[BAR ] {date} | Px={price:.6f} | RSI={rsiv:.2f} | WMA={wmav:.6f}")

        # Actualiza filtros RSI
        if rsiv < self.p.n2:
            if not self.buy_filter:
                print(f"[FILTER ON] BUY (RSI<{self.p.n2})")
            self.buy_filter, self.sell_filter = True, False
        elif rsiv > self.p.n1:
            if not self.sell_filter:
                print(f"[FILTER ON] SELL (RSI>{self.p.n1})")
            self.sell_filter, self.buy_filter = True, False

        # Stops dinÃ¡micos si hay posiciÃ³n
        if self.position.size > 0 and self.stop_long is not None and price < self.stop_long:
            print(f"[STOP] LONG hit @ {self.stop_long:.6f}")
            self.close();
            self.stop_long = None;
            return
        if self.position.size < 0 and self.stop_short is not None and price > self.stop_short:
            print(f"[STOP] SHORT hit @ {self.stop_short:.6f}")
            self.close();
            self.stop_short = None;
            return

        lookback = int(self.p.n3) if self.p.n3 and self.p.n3 > 1 else 10

        # BUY filter + giro WMA arriba
        if self.buy_filter and self._wma_turns_up():
            if self.position.size < 0:
                print("[REV ] Cerrar corto -> largo")
                self.close()
            self.stop_long = self.piv.last_min
            print(f"[BUY ] WMA arriba con BUY filter | stop_long={self.stop_long:.6f}")
            self.buy(size=self.size_percent(self.p.size_pct))

        # SELL filter + giro WMA abajo
        elif self.sell_filter and self._wma_turns_down():
            if self.position.size > 0:
                print("[REV ] Cerrar largo -> corto")
                self.close()
            self.stop_short = self.piv.last_min
            print(f"[SELL] WMA abajo con SELL filter | stop_short={self.stop_short:.6f}")
            self.sell(size=self.size_percent(self.p.size_pct))

    def export_indicators(self):
        return self._export_indicators_helper(
            self.rsi, self.wma,
            names=['RSI', 'WMA']
        )

########################################################################################################################
class Bollinger_Bands(_BaseLoggedStrategy):
    """
    ReversiÃ³n a la media con Bandas de Bollinger usando CRUCES:
      - Ruptura superior = Close cruza por ENCIMA de la banda superior  (x_close_upper > 0)
      - Reentrada       = Close cruza por DEBAJO de la banda superior (x_close_upper < 0) -> CORTO
      - Ruptura inferior= Close cruza por DEBAJO de la banda inferior  (x_close_lower < 0)
      - Reentrada       = Close cruza por ENCIMA de la banda inferior (x_close_lower > 0) -> LARGO
      - Stop  = Ãºltimo pivote confirmado (mÃ¡ximo/minimo)
      - TP    = banda media
    """
    params = (('n1', 50), ('n2', 2), ('n3', 10), ('size_pct', 0.05))  # n3 no se usa aquÃ­; conservo la firma

    def __init__(self):
        # AÃ±adir esta condiciÃ³n en todas las estrategias que usen make_multi.
        if not getattr(self, '_is_multi_ctx', False):
            super().__init__()
        # --- Bandas de Bollinger ---
        self.bb = bt.talib.BBANDS(
            self.data.close,
            timeperiod=self.p.n1,
            nbdevup=float(self.p.n2),
            nbdevdn=float(self.p.n2)
        )
        # --- Pivotes (para stop) ---
        self.piv = Pivot3Candle(self.data)

        # --- Flags de â€œhubo rupturaâ€ ---
        self.close_above_upper = False
        self.close_below_lower = False

        # --- Stops y tamaÃ±o ---
        self.stop_short = None
        self.stop_long = None
        
        # --- CRUCES con bandas ---
        # x_close_upper: +1 si close cruza por ENCIMA de upper (ruptura arriba), -1 si cruza por DEBAJO (reentrada)
        self.x_close_upper = bt.indicators.CrossOver(self.data.close, self.bb.upperband)
        # x_close_lower: +1 si close cruza por ENCIMA de lower (reentrada), -1 si cruza por DEBAJO (ruptura abajo)
        self.x_close_lower = bt.indicators.CrossOver(self.data.close, self.bb.lowerband)

        print(f"[INIT] Bollinger_Bands (cruces) | n1={self.p.n1} n2={self.p.n2} size={self.p.size_pct}")

    def next(self):
        date  = self.data.datetime.datetime(0)
        price = float(self.data.close[0])
        upper = float(self.bb.upperband[0])
        mid   = float(self.bb.middleband[0])
        lower = float(self.bb.lowerband[0])

        # -------------------------------
        # 1) Detectar RUPTURAS por cruce
        # -------------------------------
        if self.x_close_upper[0] > 0:
            # Close cruzÃ³ por ENCIMA de la banda superior -> ruptura superior
            self.close_above_upper = True
            # print(f"[FLAG] {date} Ruptura superior por cruce (close â†‘ upper)")

        if self.x_close_lower[0] < 0:
            # Close cruzÃ³ por DEBAJO de la banda inferior -> ruptura inferior
            self.close_below_lower = True
            # print(f"[FLAG] {date} Ruptura inferior por cruce (close â†“ lower)")

        # --------------------------------------------
        # 2) ENTRADAS por REENTRADA (cruce inverso)
        # --------------------------------------------
        # --- CORTO: hubo ruptura superior y ahora reentrada (close cruza por DEBAJO de upper)
        if self.position.size == 0 and self.close_above_upper and self.x_close_upper[0] < 0:
            #print(f"[ENTRY SHORT] {date} Reentrada tras ruptura superior (close â†“ upper).")
            stop_price_short = self.piv.last_max # Calcular el stop ANTES sino error
            self.sell(size=self.size_percent_by_stop(self.p.size_pct, stop_price_short))
            self.stop_short = stop_price_short
            #print(f"[ENTRY SHORT] Stop fijo = {self.stop_short}")
            self.close_above_upper = False  # consumimos la flag

        # --- LARGO: hubo ruptura inferior y ahora reentrada (close cruza por ENCIMA de lower)
        if self.position.size == 0 and self.close_below_lower and self.x_close_lower[0] > 0:
            #print(f"[ENTRY LONG ] {date} Reentrada tras ruptura inferior (close â†‘ lower).")
            stop_price_long = self.piv.last_min # Calcular el stop ANTES sino error
            self.buy(size=self.size_percent_by_stop(self.p.size_pct, stop_price_long))
            self.stop_long = stop_price_long
            #print(f"[ENTRY LONG ] Stop fijo = {self.stop_long}")
            self.close_below_lower = False  # consumimos la flag
        
        # ---------------------
        # 3) SALIDAS: SHORT
        # ---------------------
        if self.position.size < 0:
            # STOP por invalidaciÃ³n (precio supera el Ãºltimo mÃ¡ximo confirmado)
            if self.stop_short is not None and price > self.stop_short:
                #print(f"[EXIT SHORT - STOP] {date} Precio > stop ({self.stop_short}).")
                self.close()
                self.stop_short = None
            # TP en la banda media
            elif price < mid:
                #print(f"[EXIT SHORT - TP] {date} Precio < banda media ({mid}).")
                self.close()
                self.stop_short = None

        # ---------------------
        # 4) SALIDAS: LONG
        # ---------------------
        if self.position.size > 0:
            # STOP por invalidaciÃ³n (precio cae por debajo del Ãºltimo mÃ­nimo confirmado)
            if self.stop_long is not None and price < self.stop_long:
                #print(f"[EXIT LONG - STOP] {date} Precio < stop ({self.stop_long}).")
                self.close()
                self.stop_long = None
            # TP en la banda media
            elif price > mid:
                #print(f"[EXIT LONG - TP] {date} Precio > banda media ({mid}).")
                self.close()
                self.stop_long = None

    def export_indicators(self):
        return self._export_indicators_helper(
            self.bb.upperband, self.bb.middleband, self.bb.lowerband,
            self.piv.last_pivot_max, self.piv.last_pivot_min,
            names=['BB_Upper', 'BB_Middle', 'BB_Lower', 'PIVOT_LAST_MAX', 'PIVOT_LAST_MIN']
        )

########################################################################################################################
class Bollinger_Ratio(_BaseLoggedStrategy):
    """
    Estrategia multi-activo sobre el RATIO (datas[2]) con Bollinger:
      - Ruptura superior + reentrada -> SHORT base + LONG quote.
      - Ruptura inferior + reentrada -> LONG base + SHORT quote.
      - Salidas: stop en pivotes y TP en banda media del ratio.
    Requiere:
      datas[0] = base, datas[1] = quote, datas[2] = ratio (para seÃ±ales).
    """
    params = (('n1', 40), ('n2', 1), ('n3', 10), ('size_pct', 0.45))  

    def __init__(self):
        super().__init__()
        self.ratio = self.datas[0]
        self.base = self.datas[1]
        self.quote = self.datas[2]
       
        #self.plotinfo.plotmaster = self.ratio

        self.bb = bt.talib.BBANDS(self.ratio.close, timeperiod=self.p.n1, nbdevup=float(self.p.n2), nbdevdn=float(self.p.n2))
        self.piv = Pivot3Candle(self.ratio)

        self.ratio_above_upper = False
        self.ratio_below_lower = False
        self.last_max = None
        self.last_min = None
        self.stop_short = None
        self.stop_long = None

    def next(self):
        date = self.ratio.datetime.datetime(0)
        

        ratio_price = float(self.ratio.close[0])
        ratio_high = float(self.ratio.high[0])
        ratio_low = float(self.ratio.low[0])

        upper = float(self.bb.upperband[0])    # Banda superior
        mid = float(self.bb.middleband[0])     # Banda media (SMA)
        lower = float(self.bb.lowerband[0])   # Banda inferior

        pos_base = self.getposition(self.base).size
        pos_quote = self.getposition(self.quote).size
        has_pos = (pos_base != 0 or pos_quote != 0)

        # Flags de ruptura
        if ratio_price > upper:
            self.ratio_above_upper = True
        if ratio_price < lower:
            self.ratio_below_lower = True

        # Entradas por reingreso
        if not has_pos and self.ratio_above_upper and ratio_price < upper and (self.piv.last_max is not None):
            stop_price_short = self.piv.last_max # Calcular el stop ANTES sino error
            size_base = self.size_percent_by_stop(self.p.size_pct, stop_price_short, data=self.base)
            size_quote = self.size_percent_by_stop(self.p.size_pct, stop_price_short, data=self.quote)
            self.sell(data=self.base, size=size_base)   # SHORT base
            self.buy(data=self.quote, size=size_quote)  # LONG quote
            # Stop FIJO tomado en el momento de la entrada
            self.stop_short = float(stop_price_short)
            self.ratio_above_upper = False

        if not has_pos and self.ratio_below_lower and ratio_price > lower and (self.piv.last_min is not None):
            stop_price_long = self.piv.last_min # Calcular el stop ANTES sino error
            size_base = self.size_percent_by_stop(self.p.size_pct, stop_price_long, data=self.base)
            size_quote = self.size_percent_by_stop(self.p.size_pct, stop_price_long, data=self.quote)
            self.buy(data=self.base, size=size_base)    # LONG base
            self.sell(data=self.quote, size=size_quote) # SHORT quote
            # Stop FIJO tomado en el momento de la entrada
            self.stop_long = float(stop_price_long)
            self.ratio_below_lower = False

        # Salidas: SHORT base + LONG quote
        if pos_base < 0 and pos_quote > 0:
            if self.stop_short and ratio_price > self.stop_short:
                self.close(data=self.base);
                self.close(data=self.quote);
                self.stop_short = None
            elif ratio_price < mid:
                self.close(data=self.base);
                self.close(data=self.quote);
                self.stop_short = None

        # Salidas: LONG base + SHORT quote
        if pos_base > 0 and pos_quote < 0:
            if self.stop_long and ratio_price < self.stop_long:
                self.close(data=self.base);
                self.close(data=self.quote);
                self.stop_long = None
            elif ratio_price > mid:
                self.close(data=self.base);
                self.close(data=self.quote);
                self.stop_long = None

    def export_indicators(self):
        return self._export_indicators_helper(
            self.bb.upperband, self.bb.middleband, self.bb.lowerband,
            self.piv.last_pivot_max, self.piv.last_pivot_min,
            names=['Ratio_BB_Upper', 'Ratio_BB_Middle', 'Ratio_BB_Lower', 'PIVOT_LAST_MAX', 'PIVOT_LAST_MIN']
        )

# ==========================================================================
# ESTRATEGIAS DE EVALUACION EN MULTIACTIVO BASADAS EN ESTRATEGIAS INICIALES 
# ========================================================================== 
# Idea general:
#   - StrategyMultiAsset es una estrategia genÃ©rica que ENVUELVE a una
#     estrategia "single asset" ya existente (inner_cls).
#   - Aplica SU __init__ y SU next() a cada data por separado mediante
#     un contexto interno (_InnerContext), pero usando SIEMPRE el mismo
#     broker/capital de esta clase.
#   - Resultado: una sola cuenta / curva de equity, con todas las
#     posiciones de todos los activos sumadas vela a vela (cartera).
#   - Usar make_multi(strategy_cls) para convertir una estrategia single-asset en multi-asset.
#     si pasamos los activos solo en data01 evalua activo por activo, sin evaluar la cartera completa.
# ================================================================

class StrategyMultiAsset(_BaseLoggedStrategy):
    """
    Wrapper genÃ©rico multi-activo.
    ...
    """
    
    # Atributo de clase para parÃ¡metros por activo
    _params_by_asset = None  # Dict[str, Dict[str, Any]] mapea asset_name -> params

    def __init__(self):
        """
        Constructor:
          - Inicializa la parte base (_BaseLoggedStrategy).
          - Crea un contexto interno por cada data.
          - Aplica inner_cls.__init__(ctx) sobre cada contexto (indicadores).
        """
        super().__init__()

        if self.inner_cls is None:
            raise ValueError("Debes definir inner_cls en la subclase de StrategyMultiAsset")

        # Lista de contextos, uno por cada data
        self._contexts: List["StrategyMultiAsset._InnerContext"] = []

        print("================================================================")
        print(f"[INIT] StrategyMultiAsset base={self.inner_cls.__name__}")
        print(f"       NÂº de datas detectados={len(self.datas)}")
        print("================================================================")

        # Crear un contexto interno para cada data
        for idx, data in enumerate(self.datas):
            data_name = getattr(data, "_name", f"data{idx}")
            print(f"[INIT]  -> Creando contexto interno para data[{idx}] = {data_name}")

            # Obtener asset_name desde el DataFrame original
            asset_name = None
            if hasattr(data, 'p') and hasattr(data.p, 'dataname'):
                df = data.p.dataname
                asset_name = getattr(df, '_asset_name', None)
            
            # Si no se encontrÃ³, intentar desde el nombre del datafeed
            if not asset_name:
                asset_name = getattr(data, '_asset_name', data_name)

            ctx = StrategyMultiAsset._InnerContext(
                outer=self,
                data=data,
                data_index=idx,
                asset_name=asset_name,
                params_by_asset=self._params_by_asset
            )

            # MUY IMPORTANTE:
            #   AquÃ­ reutilizamos __init__ de la estrategia original (inner_cls),
            #   pero pasando ctx como "self". Esto crea los indicadores en ctx
            #   igual que si la estrategia operara sobre un solo activo.
            ctx._is_multi_ctx = True
            self.inner_cls.__init__(ctx)

            self._contexts.append(ctx)

        # Guardamos puntero al next "desvinculado" de inner_cls
        self._inner_next = self.inner_cls.next

    def next(self):
        """
        MÃ©todo que Backtrader llama en cada nueva vela.

        AquÃ­:
          - Obtenemos la fecha global (primer data) solo para logs.
          - Recorremos todos los contextos.
          - Ejecutamos inner_cls.next(ctx) para cada activo.
        """
        if self.datas:
            dt = self.datas[0].datetime.datetime(0)
        else:
            dt = None

        #print("----------------------------------------------------------------")
        #print(f"[MULTI NEXT] {dt} | n_datas={len(self.datas)} "
        #      f"| estrategia_base={self.inner_cls.__name__}")
        #print("----------------------------------------------------------------")

        for ctx in self._contexts:
            data_name = getattr(ctx.data, "_name", f"data{ctx.data_index}")
            #print(f"[MULTI NEXT] Ejecutando lÃ³gica base en '{data_name}'")
            # inner_cls.next(ctx) -> ctx hace de â€œselfâ€ para la estrategia base
            self._inner_next(ctx)

    def export_indicators(self):
        """
        Exporta indicadores.

        Por simplicidad y compatibilidad con tu pipeline actual, devolvemos
        los indicadores del PRIMER contexto (primer activo), utilizando el
        mÃ©todo export_indicators(self) de inner_cls si existe.
        """
        if not self._contexts:
            print("[MULTI EXPORT] Sin contextos internos, no hay indicadores")
            return {}

        ctx0 = self._contexts[0]

        if hasattr(self.inner_cls, "export_indicators"):
            print("[MULTI EXPORT] Llamando a export_indicators() del primer contexto")
            return self.inner_cls.export_indicators(ctx0)

        print("[MULTI EXPORT] inner_cls no implementa export_indicators()")
        return {}

    # ------------------------------------------------------------
    # Contexto interno por data (un activo)
    # ------------------------------------------------------------
    class _InnerContext:
        """
        Contexto interno que simula ser una instancia de inner_cls para
        UN solo activo.
        """

        def __init__(self, outer: "_BaseLoggedStrategy", data, data_index: int, 
                    asset_name: str = None, params_by_asset: dict = None):
            self._outer = outer
            self.data = data
            self.datas = [data]
            self.data_index = data_index
            self._asset_name = asset_name
            self._inner_cls = outer.inner_cls  # Guardar referencia a la clase interna

            # Compartimos broker con la estrategia externa
            self.broker = outer.broker

            # Calcular nÃºmero total de activos para rebalanceo equitativo
            num_assets = len(outer.datas) if hasattr(outer, 'datas') else 1

            # ParÃ¡metros: usar especÃ­ficos del activo si estÃ¡n disponibles, sino usar los de outer
            if params_by_asset and asset_name and asset_name in params_by_asset:
                # Crear objeto Params con los parÃ¡metros especÃ­ficos del activo
                asset_params = params_by_asset[asset_name]
                # Combinar con parÃ¡metros base de outer.p
                combined_params = {}
                # Primero copiar todos los parÃ¡metros base
                for key in dir(outer.p):
                    if not key.startswith('_'):
                        try:
                            combined_params[key] = getattr(outer.p, key)
                        except:
                            pass
                # Luego sobrescribir con parÃ¡metros especÃ­ficos del activo
                combined_params.update(asset_params)
                
                # Rebalancear size_pct equitativamente: dividir por nÃºmero de activos
                if num_assets > 1 and 'size_pct' in combined_params:
                    base_size_pct = combined_params['size_pct']
                    rebalanced_size_pct = base_size_pct / num_assets
                    combined_params['size_pct'] = rebalanced_size_pct
                    print(f"[CTX INIT] size_pct rebalanceado para {asset_name}: {base_size_pct:.4f} -> {rebalanced_size_pct:.4f} (N={num_assets} activos)")
                elif num_assets > 1:
                    # Si no hay size_pct en asset_params, usar el de outer.p
                    base_size_pct = getattr(outer.p, 'size_pct', 0.05)
                    rebalanced_size_pct = base_size_pct / num_assets
                    combined_params['size_pct'] = rebalanced_size_pct
                    print(f"[CTX INIT] size_pct rebalanceado para {asset_name}: {base_size_pct:.4f} -> {rebalanced_size_pct:.4f} (N={num_assets} activos, usando base)")
                
                # Crear objeto Params personalizado
                class AssetParams:
                    def __init__(self, params_dict):
                        for k, v in params_dict.items():
                            setattr(self, k, v)
                
                self.p = AssetParams(combined_params)
                print(f"[CTX INIT] ParÃ¡metros especÃ­ficos para {asset_name}: {asset_params}")
            else:
                # Usar parÃ¡metros compartidos (comportamiento original)
                self.p = outer.p
                
                # Aplicar rebalanceo equitativo incluso si no hay params_by_asset
                if num_assets > 1 and hasattr(self.p, 'size_pct'):
                    base_size_pct = self.p.size_pct
                    rebalanced_size_pct = base_size_pct / num_assets
                    # Crear copia modificada de los params
                    class RebalancedParams:
                        def __init__(self, base_p, size_pct_val):
                            for key in dir(base_p):
                                if not key.startswith('_'):
                                    try:
                                        setattr(self, key, getattr(base_p, key))
                                    except:
                                        pass
                            self.size_pct = size_pct_val
                    self.p = RebalancedParams(outer.p, rebalanced_size_pct)
                    print(f"[CTX INIT] size_pct rebalanceado (sin params_by_asset): {base_size_pct:.4f} -> {rebalanced_size_pct:.4f} (N={num_assets} activos)")

        # ------------------- Estado de la posiciÃ³n -------------------

        @property
        def position(self):
            """
            Devuelve la posiciÃ³n ESPECÃFICA de este activo usando el broker
            de la estrategia externa (outer).
            """
            pos = self._outer.getposition(self.data)
            size = float(getattr(pos, "size", 0.0)) if pos is not None else 0.0
            data_name = getattr(self.data, "_name", f"data{self.data_index}")
            # print(f"[CTX POS] data={data_name} idx={self.data_index} pos.size={size}")
            return pos

        # ------------------- Helpers de tamaÃ±o -------------------

        def size_percent(self, pct: float, *args, **kwargs) -> float:
            """
            Delegamos en size_percent de outer, fijando data=este activo.
            """
            size = self._outer.size_percent(pct, data=self.data, *args, **kwargs)
            data_name = getattr(self.data, "_name", f"data{self.data_index}")
            # print(f"[CTX SIZE] data={data_name} pct={pct} -> size={size}")
            return size

        def size_percent_by_stop(self, risk_pct: float, stop_price: float, *args, **kwargs) -> float:
            """
            Delegamos en size_percent_by_stop de outer, fijando data=este activo.
            """
            size = self._outer.size_percent_by_stop(
                risk_pct,
                stop_price,
                data=self.data,
                *args,
                **kwargs,
            )
            data_name = getattr(self.data, "_name", f"data{self.data_index}")
            print(
                f"[CTX SIZE STOP] data={data_name} "
                f"risk_pct={risk_pct} stop={stop_price} -> size={size}"
            )
            return size

        # -------------------------- Ã“rdenes ------------------------------

        def buy(self, *args, **kwargs):
            """
            Lanza una orden de compra en ESTE activo, delegando en outer.buy().
            """
            data_name = getattr(self.data, "_name", f"data{self.data_index}")
            # print(f"[CTX BUY] data={data_name}")
            return self._outer.buy(data=self.data, *args, **kwargs)

        def sell(self, *args, **kwargs):
            """
            Lanza una orden de venta (corto) en ESTE activo, delegando en outer.sell().
            """
            data_name = getattr(self.data, "_name", f"data{self.data_index}")
            # print(f"[CTX SELL] data={data_name}")
            return self._outer.sell(data=self.data, *args, **kwargs)

        def close(self, *args, **kwargs):
            """
            Cierra la posiciÃ³n de ESTE activo, delegando en outer.close().
            """
            data_name = getattr(self.data, "_name", f"data{self.data_index}")
            # print(f"[CTX CLOSE] data={data_name}")
            return self._outer.close(data=self.data, *args, **kwargs)

        # ------------------ Utilidades auxiliares ------------------------

        def _export_indicators_helper(self, *indicators, names=None):
            """
            Permite que inner_cls.export_indicators(ctx) reutilice el helper
            de la estrategia externa para exportar indicadores.
            """
            return self._outer._export_indicators_helper(*indicators, names=names)

        def notify_order(self, order):
            """
            Si inner_cls llega a llamar a notify_order, reenviamos a outer.
            """
            return self._outer.notify_order(order)

        def notify_trade(self, trade):
            """
            Si inner_cls llega a llamar a notify_trade, reenviamos a outer.
            """
            return self._outer.notify_trade(trade)

        def __getattr__(self, name):
            """
            Delegar mÃ©todos que no estÃ¡n en _InnerContext hacia inner_cls.
            Esto permite que mÃ©todos auxiliares como _wma_turns_up() funcionen
            cuando se ejecutan desde next().
            Solo se activa si el atributo no existe en _InnerContext.
            """
            if hasattr(self._inner_cls, name):
                method = getattr(self._inner_cls, name)
                if callable(method):
                    # Crear un mÃ©todo vinculado que use 'self' (el contexto) como primer argumento
                    def bound_method(*args, **kwargs):
                        return method(self, *args, **kwargs)
                    return bound_method
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


def make_multi(strategy_cls):
    """
    Envuelve una estrategia 'single-asset' en versiÃ³n multi-activo (StrategyMultiAsset),
    preservando correctamente los params aunque vengan como objeto Backtrader Params.
    """
    if issubclass(strategy_cls, StrategyMultiAsset):
        return strategy_cls  # ya es multi

    def _extract_bt_params(pobj):
        # Intenta extraer nombres/valores de un objeto Params de Backtrader
        try:
            keys = pobj._getkeys()
            return tuple((k, getattr(pobj, k)) for k in keys)
        except Exception:
            return ()

    base_params = getattr(strategy_cls, 'params', ())
    if isinstance(base_params, (tuple, list)):
        try:
            wrapper_params = tuple(tuple(p) for p in base_params)
        except TypeError:
            wrapper_params = ()
    else:
        wrapper_params = _extract_bt_params(base_params)

    class _AutoMulti(StrategyMultiAsset):
        inner_cls = strategy_cls
        params = wrapper_params

    _AutoMulti.__name__ = f"{strategy_cls.__name__}"
    return _AutoMulti


def make_multi_tf(strategy_cls):
    """
    Envuelve una estrategia multi-timeframe para evitar sincronizaciÃ³n de DataFrames
    en main.py, preservando el acceso normal a self.datas[0], self.datas[1], etc.
    
    USO:
        from strategies import PivotZoneTest, make_multi_tf
        
        STRATEGIES = [
            make_multi_tf(PivotZoneTest)  # Evita sincronizaciÃ³n
        ]
    
    DIFERENCIA CON make_multi:
        - make_multi: para mÃºltiples ACTIVOS (crea contextos separados)
        - make_multi_tf: para mÃºltiples TIMEFRAMES del mismo activo (sin contextos)
    
    VENTAJAS:
        - Preserva todas las velas del timeframe de entrada (data0)
        - Backtrader sincroniza automÃ¡ticamente los diferentes timeframes
        - Transparente para la estrategia (funciona igual que sin wrapper)
    """
    if issubclass(strategy_cls, StrategyMultiAsset):
        return strategy_cls  # ya es multi
    
    def _extract_bt_params(pobj):
        # Intenta extraer nombres/valores de un objeto Params de Backtrader
        try:
            keys = pobj._getkeys()
            return tuple((k, getattr(pobj, k)) for k in keys)
        except Exception:
            return ()

    base_params = getattr(strategy_cls, 'params', ())
    if isinstance(base_params, (tuple, list)):
        try:
            wrapper_params = tuple(tuple(p) for p in base_params)
        except TypeError:
            wrapper_params = ()
    else:
        wrapper_params = _extract_bt_params(base_params)
    
    # Crear una nueva clase que hereda SOLO de la estrategia original
    # con un atributo especial para marcar como multi-timeframe
    class _MultiTimeframeWrapper(strategy_cls):
        """
        Wrapper para estrategias multi-timeframe.
        Marca la estrategia con _is_multi_tf para que main.py evite sincronizaciÃ³n,
        pero mantiene toda la funcionalidad original sin crear contextos separados.
        """
        inner_cls = strategy_cls
        params = wrapper_params
        # Atributo especial para marcar como multi-timeframe
        _is_multi_tf = True
        
        def __init__(self):
            print("=" * 70)
            print(f"[INIT] MultiTimeframe Wrapper: {strategy_cls.__name__}")
            print(f"       NÃºmero de datas: {len(self.datas)}")
            print("       Modo: Multi-timeframe (sin sincronizaciÃ³n previa)")
            print("=" * 70)
            
            # Llamar al __init__ de la estrategia original
            super().__init__()
    
    _MultiTimeframeWrapper.__name__ = f"{strategy_cls.__name__}"
    return _MultiTimeframeWrapper

