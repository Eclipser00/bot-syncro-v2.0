"""
Config unico del bot de trading. Este archivo es la fuente de verdad para todas
las configuraciones y es consumido directamente por `bot_trading/main.py`.

Como usar esta configuracion:
1) Cambia ACTIVE_ENV a "development", "production" o "testing".
2) Ajusta las secciones Broker/Activos/Estrategias/Riesgo/Logs segun tus
   necesidades; main.py leera siempre estos valores.
3) validate_config(settings) se ejecuta antes de arrancar el bot para asegurar
   coherencia.

Todo el codigo debe importar configuracion desde aqui:
    from config import settings, validate_config

No se usa .env a menos que config.broker.load_env_credentials sea True.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
import sys
import subprocess


def _reexec_with_project_venv_if_needed() -> None:
    """Relanza este config.py con el .venv local si VSCode usa otro interprete."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    local_python = os.path.join(project_dir, ".venv", "Scripts", "python.exe")
    if not os.path.isfile(local_python):
        return

    current_python = os.path.abspath(sys.executable)
    target_python = os.path.abspath(local_python)
    if current_python.lower() == target_python.lower():
        return

    marker_name = "BOT_TRADING_CONFIG_VENV_REEXEC_DONE"
    if os.environ.get(marker_name) == "1":
        return

    print(f"[config.py] Interprete detectado: {current_python}")
    print(f"[config.py] Reintentando con venv local: {target_python}")
    env = os.environ.copy()
    env[marker_name] = "1"
    result = subprocess.run([target_python, os.path.abspath(__file__), *sys.argv[1:]], env=env)
    raise SystemExit(result.returncode)


# =============================================================================
# 1) Toggles principales
# =============================================================================

# Selecciona el entorno activo: "development" (paper/FakeBroker), "production" (MT5 real) o "testing".
ACTIVE_ENV = "development"  # entorno que usara el bot al arrancar


# =============================================================================
# 2) Bloques de configuracion
# =============================================================================


@dataclass
class BrokerConfig:
    """Parametros del broker.

    use_real_broker: True envia ordenes a MetaTrader5 real; False usa FakeBroker (paper).
    max_retries / retry_delay: reintentos de conexion/envio (unidades: intentos, segundos).
    load_env_credentials: si True, toma server/login/password de variables de entorno con prefijo env_prefix.
    server/login/password: credenciales MT5 (no guardar valores reales en git; usa .env).
    """

    use_real_broker: bool  # True envia ordenes a MT5 real; False usa FakeBroker
    max_retries: int  # reintentos maximos al fallar conexion/envio
    retry_delay: float  # segundos de espera entre reintentos
    load_env_credentials: bool = False  # si es True, lee credenciales desde variables de entorno
    env_prefix: str = "MT5_"  # prefijo usado al buscar credenciales en entorno
    server: Optional[str] = None  # host o servidor MT5
    login: Optional[str] = None  # numero de cuenta MT5
    password: Optional[str] = None  # password de la cuenta MT5


@dataclass
class LoggingConfig:
    """Ajustes de logging.

    level: DEBUG/INFO/WARNING/ERROR.
    log_to_file: True escribe a disco; False solo consola.
    log_file_path: ruta del archivo de log (relativa a la raiz del repo).
    pivot_log_file_path: ruta del log dedicado de zonas pivote (relativa a la raiz).
    format: formato de linea de log.
    """

    level: str  # nivel minimo de log (DEBUG/INFO/WARNING/ERROR)
    log_to_file: bool  # True escribe en archivo ademas de consola
    log_file_path: Optional[str] = None  # ruta del archivo de log si se usa
    pivot_log_file_path: Optional[str] = "logs/pivot_zones.log"  # ruta del log dedicado de zonas pivote
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # formato de linea de log


@dataclass
class RiskConfig:
    """Limites de riesgo (porcentaje, sin signo).

    dd_global: drawdown maximo global del bot.
    dd_por_activo: drawdown maximo por simbolo.
    dd_por_estrategia: drawdown maximo por estrategia.
    initial_balance: balance de referencia para calculos (moneda de la cuenta).
    max_margin_usage_percent: porcentaje maximo del capital que se permite usar como margen.
    """

    dd_global: float  # drawdown maximo permitido a nivel bot (%)
    dd_por_activo: Dict[str, float] = field(default_factory=dict)  # drawdown maximo por simbolo (%)
    dd_por_estrategia: Dict[str, float] = field(default_factory=dict)  # drawdown maximo por estrategia (%)
    initial_balance: float = 20_000.0  # balance de referencia para calculos
    max_margin_usage_percent: float = 80.0  # porcentaje maximo de margen a usar


@dataclass
class SymbolConfigEntry:
    """Activos a operar.

    name: simbolo en el broker (ej: EURUSD).
    min_timeframe: timeframe minimo disponible en el broker (ej: M1).
    n1/n2/n3/size_pct/p: overrides opcionales por simbolo para estrategias.
    """

    name: str  # simbolo tal como lo expone el broker (p.ej. EURUSD)
    min_timeframe: str  # timeframe minimo disponible en broker
    n1: Optional[int] = None  # override de lookback ATR (velas) por simbolo
    n2: Optional[int] = None  # override de multiplicador % ATR por simbolo
    n3: Optional[int] = None  # override de pivotes requeridos por simbolo
    size_pct: Optional[float] = None  # override de size_pct por simbolo
    p: Optional[float] = None  # override de multiplicador de pivote por simbolo


@dataclass
class StrategyConfig:
    """Parametros de estrategia (PivotZoneTest en este repo).

    name: identificador unico (usado para logs y magic number).
    allowed_symbols: lista blanca de simbolos. Si se deja vacia/None, se usan
        todos los simbolos definidos en `symbols`.
    tf_entry/tf_zone/tf_stop: timeframes usados por la logica.
    n1: periodos ATR para ancho de zona (velas TF_zone).
    n2: multiplicador (%) aplicado al ATR para definir ancho de zona.
    n3: cantidad de pivotes necesarios para bloquear una zona.
    size_pct: porcentaje del balance para calcular el lote (0.05 = 5%).
    p: multiplicador de pivote, impacto directo en niveles de zona.
    timeframes: lista de timeframes que el motor debe descargar.
    """

    name: str  # identificador unico de la estrategia
    allowed_symbols: Optional[List[str]]  # lista blanca de simbolos permitidos
    tf_entry: str  # timeframe que usa la logica de entrada
    tf_zone: str  # timeframe para calcular zonas pivote
    tf_stop: str  # timeframe que define stops
    n1: int  # lookback ATR (velas) para calcular ancho de zona
    n2: int  # multiplicador (%) aplicado al ATR para ancho de zona
    n3: int  # pivotes requeridos para bloquear una zona
    size_pct: float  # porcentaje del balance para dimensionar la posicion
    p: float  # multiplicador de pivote/zona
    timeframes: List[str] = field(default_factory=list)  # timeframes que debe descargar el motor


@dataclass
class LoopConfig:
    """Parametros del bucle de ejecucion sincronizado.

    timeframe_minutes: periodo base de la vela en minutos (1=M1, 5=M5...).
    wait_after_close: segundos de espera tras cerrar la vela antes de ejecutar el ciclo.
    skip_sleep_when_simulated: si True y hay reloj simulado (CSV), no se usa sleep y se avanza por timestamps.
        Por defecto es False para mantener el mismo flujo que produccion.
    """

    timeframe_minutes: int = 1  # tamano de vela base en minutos
    wait_after_close: int = 5  # segundos de espera tras cierre de vela
    skip_sleep_when_simulated: bool = True  # omite sleep cuando se usan datos simulados CSV


@dataclass
class DataConfig:
    """Config de feeds/series.

    """

    data_mode: str = "production"  # origen de datos: production o development
    data_development_dir: str = "data_development"  # carpeta con CSV de desarrollo
    bootstrap_lookback_days_zone: int = 14  # dias hacia atras para inicializar zonas
    bootstrap_lookback_days_entry: Optional[int] = None  # override para lookback de entradas
    bootstrap_lookback_days_stop: Optional[int] = None  # override para lookback de stops
    csv_base_timeframe: str = "M1"  # timeframe base para CSV de desarrollo


@dataclass
class BotConfig:
    """Config global unificada del bot."""

    name: str  # nombre del bot para logs y magic numbers
    mode: str  # modo de ejecucion (production/development/testing)
    broker: BrokerConfig  # configuracion del broker o FakeBroker
    logging: LoggingConfig  # configuracion de logging
    risk: RiskConfig  # limites y parametros de riesgo
    symbols: List[SymbolConfigEntry]  # simbolos disponibles para operar
    strategies: List[StrategyConfig]  # estrategias activas
    loop: LoopConfig  # parametros del bucle de ejecucion
    data: DataConfig  # configuracion de feeds/datos


# =============================================================================
# 3) Perfiles (production/development/testing)
# =============================================================================

# Configuracion base compartida: se reutiliza en los tres entornos para mantener
# el flujo alineado. Las diferencias entre entornos se limitan a broker/datos/
# logging/riesgo; las estrategias y simbolos comparten la misma declaracion.
DEFAULT_SYMBOLS: List[SymbolConfigEntry] = [
    SymbolConfigEntry(
        name="EURUSD",
        min_timeframe="M1",
        n1=3,
        n2=100,
        n3=5,
        size_pct=0.05,
        p=0.50,
    ),
    SymbolConfigEntry(
        name="GBPUSD",
        min_timeframe="M1",
        n1=3,
        n2=100,
        n3=5,
        size_pct=0.05,
        p=0.50,
    ),
    SymbolConfigEntry(
        name="USDJPY",
        min_timeframe="M1",
        n1=3,
        n2=100,
        n3=5,
        size_pct=0.05,
        p=0.50,
    ),
]

DEFAULT_STRATEGIES: List[StrategyConfig] = [
    StrategyConfig(
        name="PivotZoneTest",
        allowed_symbols=[sym.name for sym in DEFAULT_SYMBOLS],  # se recalibra en validate_config si se agregan simbolos
        tf_entry="M1",
        tf_zone="M3",
        tf_stop="M1",
        n1=3,
        n2=100,
        n3=5,
        size_pct=0.05,
        p=0.50,
        timeframes=["M1", "M3"],
    )
]

ENVIRONMENTS: Dict[str, BotConfig] = {
    ################################################
    #             PRODUCTION CONFIG                #
    ################################################
    "production": BotConfig(
        name="last_trading_bot",  # nombre del perfil de produccion
        mode="production",  # modo de ejecucion en vivo
        broker=BrokerConfig(
            use_real_broker=True,  # usa conexion MT5 real
            max_retries=3,  # reintentos de conexion/orden antes de fallar
            retry_delay=1.0,  # segundos entre reintentos
            load_env_credentials=True,  # toma credenciales desde variables de entorno
        ),
        logging=LoggingConfig(
            level="INFO",  # nivel de log en produccion
            log_to_file=True,  # escribe logs a disco
            log_file_path="logs/production.log",  # archivo de log en produccion
            pivot_log_file_path="logs/pivot_zones.log",  # log dedicado de zonas pivote
        ),
        risk=RiskConfig(
            dd_global=30.0,  # drawdown maximo global permitido
            dd_por_activo={sym.name: 30.0 for sym in DEFAULT_SYMBOLS},
            dd_por_estrategia={
                "PivotZoneTest": 30.0,  # drawdown maximo por estrategia
            },
        ),
        symbols=copy.deepcopy(DEFAULT_SYMBOLS),
        strategies=copy.deepcopy(DEFAULT_STRATEGIES),
        loop=LoopConfig(
            timeframe_minutes=1,  # vela base en minutos
            wait_after_close=5,  # segundos de espera tras cierre
            skip_sleep_when_simulated=False,  # en vivo se respeta el sleep
        ),
        data=DataConfig(
            data_mode="production",  # usa feed del broker real
            data_development_dir="data_development",  # carpeta para CSV (no usada en prod)
            bootstrap_lookback_days_zone=15,  # dias para inicializar zonas
            bootstrap_lookback_days_entry=None,  # hereda zona si None
            bootstrap_lookback_days_stop=None,  # hereda zona si None
            csv_base_timeframe="M1",  # timeframe base de CSV
        ),
    ),
    ################################################
    #            DEVELOPMENT CONFIG                #
    ################################################
    "development": BotConfig(
        name="last_trading_bot_dev",  # nombre del perfil de desarrollo
        mode="development",  # modo de ejecucion en paper
        broker=BrokerConfig(
            use_real_broker=False,  # usa FakeBroker para paper trading
            max_retries=3,  # reintentos de conexion/orden
            retry_delay=0.1,  # segundos entre reintentos rapidos
            load_env_credentials=False,  # no carga credenciales desde entorno
        ),
        logging=LoggingConfig(
            level="DEBUG",  # logs detallados en desarrollo
            log_to_file=True,  # escribe a disco
            log_file_path="logs/development.log",  # archivo de log de desarrollo
            pivot_log_file_path="logs/pivot_zones.log",  # log dedicado de zonas pivote
        ),
        risk=RiskConfig(
            dd_global=30.0,  # drawdown maximo global
            dd_por_activo={sym.name: 30.0 for sym in DEFAULT_SYMBOLS},
            dd_por_estrategia={
                "PivotZoneTest": 30.0,  # drawdown maximo por estrategia
            },
        ),
        symbols=copy.deepcopy(DEFAULT_SYMBOLS),
        strategies=copy.deepcopy(DEFAULT_STRATEGIES),
        loop=LoopConfig(
            timeframe_minutes=1,  # vela base en minutos
            wait_after_close=5,  # segundos de espera tras cierre
            skip_sleep_when_simulated=True,  # omite sleep al simular
        ),
        data=DataConfig(
            data_mode="development",  # usa datos simulados/CSV
            data_development_dir="data_development",  # carpeta con CSV de desarrollo
            bootstrap_lookback_days_zone=0,  # dias para inicializar zonas
            bootstrap_lookback_days_entry=None,  # hereda zona si None
            bootstrap_lookback_days_stop=None,  # hereda zona si None
            csv_base_timeframe="M1",  # timeframe base de CSV
        ),
    ),
    ################################################
    #                TESTING CONFIG                #
    ################################################
    "testing": BotConfig(
        name="last_trading_bot_test",  # nombre del perfil de testing
        mode="testing",  # modo de pruebas rapidas
        broker=BrokerConfig(
            use_real_broker=False,  # broker simulado
            max_retries=1,  # un solo intento en pruebas
            retry_delay=0.0,  # sin espera entre reintentos
            load_env_credentials=False,  # no lee credenciales de entorno
        ),
        logging=LoggingConfig(
            level="DEBUG",  # logs detallados en test
            log_to_file=False,  # solo consola
            log_file_path=None,  # sin archivo de log
            pivot_log_file_path="logs/pivot_zones.log",  # log dedicado de zonas pivote
        ),
        risk=RiskConfig(
            dd_global=100.0,  # limite amplio para pruebas
            dd_por_activo={},  # sin limites por simbolo en tests
            dd_por_estrategia={},  # sin limites por estrategia en tests
        ),
        symbols=copy.deepcopy(DEFAULT_SYMBOLS),
        strategies=copy.deepcopy(DEFAULT_STRATEGIES),
        loop=LoopConfig(
            timeframe_minutes=1,  # vela base en minutos
            wait_after_close=0,  # sin espera tras cierre en tests
            skip_sleep_when_simulated=True,  # omite sleep al simular
        ),
        data=DataConfig(
            data_mode="development",  # usa datos simulados
            data_development_dir="data_development",  # carpeta con CSV de desarrollo
            bootstrap_lookback_days_zone=0,  # dias para inicializar zonas
            bootstrap_lookback_days_entry=None,  # hereda zona si None
            bootstrap_lookback_days_stop=None,  # hereda zona si None
            csv_base_timeframe="M1",  # timeframe base de CSV
        ),
    ),
}


# =============================================================================
# 4) Interfaz publica
# =============================================================================

def load_settings(env: Optional[str] = None) -> BotConfig:
    """Devuelve una copia desacoplada de la configuracion del entorno solicitado."""
    env_key = env or ACTIVE_ENV
    if env_key not in ENVIRONMENTS:
        raise ValueError(f"Entorno no soportado: {env_key}")
    return copy.deepcopy(ENVIRONMENTS[env_key])


settings: BotConfig = load_settings()


def get_config() -> BotConfig:
    """Mantiene compatibilidad: retorna la configuracion activa."""
    return settings


# =============================================================================
# 5) Validacion
# =============================================================================

def _maybe_load_broker_env(cfg: BrokerConfig) -> None:
    """Carga credenciales desde entorno si se habilita. No persiste secretos."""
    if not cfg.load_env_credentials:
        return
    prefix = cfg.env_prefix
    cfg.server = cfg.server or os.getenv(f"{prefix}SERVER")
    cfg.login = cfg.login or os.getenv(f"{prefix}LOGIN")
    cfg.password = cfg.password or os.getenv(f"{prefix}PASSWORD")


def validate_config(cfg: BotConfig) -> None:
    """Valida coherencia basica antes de arrancar."""
    _maybe_load_broker_env(cfg.broker)
    errors: List[str] = []

    if ACTIVE_ENV not in ENVIRONMENTS:
        errors.append(f"ACTIVE_ENV debe ser uno de {list(ENVIRONMENTS.keys())}")

    if not cfg.symbols:
        errors.append("Debe haber al menos un simbolo configurado.")
    symbol_names = [sym.name for sym in cfg.symbols]
    if len(symbol_names) != len(set(symbol_names)):
        errors.append("Los simbolos configurados deben ser unicos.")

    for sym in cfg.symbols:
        if not sym.name:
            errors.append("Simbolo sin nombre.")
        if not sym.min_timeframe:
            errors.append(f"Simbolo {sym.name} sin min_timeframe.")
        if sym.n1 is not None and sym.n1 <= 0:
            errors.append(f"Simbolo {sym.name} con n1 <= 0.")
        if sym.n2 is not None and sym.n2 <= 0:
            errors.append(f"Simbolo {sym.name} con n2 <= 0.")
        if sym.n3 is not None and sym.n3 <= 0:
            errors.append(f"Simbolo {sym.name} con n3 <= 0.")
        if sym.size_pct is not None and (sym.size_pct <= 0 or sym.size_pct > 1):
            errors.append(f"Simbolo {sym.name} size_pct debe estar en (0,1].")
        if sym.p is not None and sym.p <= 0:
            errors.append(f"Simbolo {sym.name} p debe ser > 0.")

    if cfg.risk.dd_global is not None and cfg.risk.dd_global <= 0:
        errors.append("dd_global debe ser > 0.")
    if cfg.risk.max_margin_usage_percent <= 0 or cfg.risk.max_margin_usage_percent > 100:
        errors.append("max_margin_usage_percent debe estar en (0, 100].")
    for k, v in cfg.risk.dd_por_activo.items():
        if v <= 0:
            errors.append(f"dd_por_activo[{k}] debe ser > 0.")
    for k, v in cfg.risk.dd_por_estrategia.items():
        if v <= 0:
            errors.append(f"dd_por_estrategia[{k}] debe ser > 0.")

    for strat in cfg.strategies:
        if strat.name == "":
            errors.append("Estrategia sin nombre.")
        if not strat.allowed_symbols:
            strat.allowed_symbols = list(symbol_names)
        else:
            unknown = [s for s in strat.allowed_symbols if s not in symbol_names]
            if unknown:
                errors.append(f"Estrategia {strat.name} tiene allowed_symbols desconocidos: {unknown}")
        if strat.tf_entry not in strat.timeframes:
            errors.append(f"Estrategia {strat.name} debe incluir tf_entry en timeframes.")
        if strat.tf_zone not in strat.timeframes:
            errors.append(f"Estrategia {strat.name} debe incluir tf_zone en timeframes.")
        if strat.tf_stop not in strat.timeframes:
            errors.append(f"Estrategia {strat.name} debe incluir tf_stop en timeframes.")
        if strat.n1 <= 0 or strat.n2 <= 0 or strat.n3 <= 0:
            errors.append(f"Estrategia {strat.name} requiere n1/n2/n3 > 0.")
        if strat.size_pct <= 0 or strat.size_pct > 1:
            errors.append(f"Estrategia {strat.name} size_pct debe estar en (0,1].")
        if strat.p <= 0:
            errors.append(f"Estrategia {strat.name} p debe ser > 0.")

    if cfg.loop.timeframe_minutes <= 0:
        errors.append("timeframe_minutes debe ser > 0.")
    if cfg.loop.wait_after_close < 0:
        errors.append("wait_after_close no puede ser negativo.")
    if not isinstance(cfg.loop.skip_sleep_when_simulated, bool):
        errors.append("skip_sleep_when_simulated debe ser booleano.")

    # Normalizar lookbacks de data (entry/stop usan zone si no se definen)
    if cfg.data.data_mode not in {"production", "development"}:
        errors.append("data_mode debe ser 'production' o 'development'.")
    if cfg.data.bootstrap_lookback_days_entry is None:
        cfg.data.bootstrap_lookback_days_entry = cfg.data.bootstrap_lookback_days_zone
    if cfg.data.bootstrap_lookback_days_stop is None:
        cfg.data.bootstrap_lookback_days_stop = cfg.data.bootstrap_lookback_days_zone
    # Coherencia entre broker real y modo de datos para evitar toggles inconsistentes
    if cfg.broker.use_real_broker and cfg.data.data_mode != "production":
        errors.append("Para broker real, data_mode debe ser 'production'.")
    if not cfg.broker.use_real_broker and cfg.data.data_mode != "development":
        errors.append("Para broker simulado, data_mode debe ser 'development'.")

    if errors:
        joined = "; ".join(errors)
        raise ValueError(f"Config invalida: {joined}")


# =============================================================================
# 6) Ejecucion directa (opcional)
# =============================================================================
if __name__ == "__main__":
    _reexec_with_project_venv_if_needed()

    # Forzar cwd al directorio del proyecto para que cualquier ruta relativa
    # se resuelva siempre contra este proyecto, aunque se ejecute desde otra carpeta.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Importar y ejecutar el entrypoint principal para permitir lanzar el bot
    # directamente desde este archivo de configuracion.
    from bot_trading import main as bot_main

    bot_main.main()
