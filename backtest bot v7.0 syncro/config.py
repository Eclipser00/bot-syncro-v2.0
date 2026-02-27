# config.py
# ================================================================
# Configuración centralizada para Backtest Bot v6.0
# ================================================================

import os
import sys
import subprocess


def _reexec_with_project_venv_if_needed() -> None:
    """Relanza este config.py con el .venv local si VSCode usa otro interprete."""
    if __name__ != "__main__":
        return

    project_dir = os.path.dirname(os.path.abspath(__file__))
    local_python = os.path.join(project_dir, ".venv", "Scripts", "python.exe")
    if not os.path.isfile(local_python):
        return

    current_python = os.path.abspath(sys.executable)
    target_python = os.path.abspath(local_python)
    if current_python.lower() == target_python.lower():
        return

    marker_name = "BACKTEST_CONFIG_VENV_REEXEC_DONE"
    if os.environ.get(marker_name) == "1":
        return

    print(f"[config.py] Interprete detectado: {current_python}")
    print(f"[config.py] Reintentando con venv local: {target_python}")
    env = os.environ.copy()
    env[marker_name] = "1"
    result = subprocess.run([target_python, os.path.abspath(__file__), *sys.argv[1:]], env=env)
    raise SystemExit(result.returncode)


_reexec_with_project_venv_if_needed()

import numpy as np

# ==================== ESTRATEGIAS ====================
# Importar las estrategias que quieras usar
# Descomenta/comenta para activar/desactivar estrategias
from strategies import (
    DosMedias,
    Tres_Medias_Filtered,
    RSI_WMA_TurnFilters,
    SMA_sobre_RSI,
    Bollinger_Bands,
    Bollinger_Ratio,
    PivotZoneTest,
    make_multi,
    make_multi_tf,
)

STRATEGIES = [
    
    # DosMedias,
    # Tres_Medias_Filtered,
    # make_multi(RSI_WMA_TurnFilters),
    # make_multi(SMA_sobre_RSI),
    # make_multi(Bollinger_Bands),
    # Bollinger_Ratio,
    PivotZoneTest  # Multi-timeframe wrapper (evita sincronización)
]

# ==================== OPTIMIZACIÓN ====================
param_cols = ['n1', 'n2', 'n3']

# Rangos de búsqueda para cada parámetro (REDUCIDOS PARA PRUEBA RÁPIDA)
# np.arange(inicio, fin, paso) genera: [inicio, inicio+paso, ..., fin-paso]
# BB
# N1_RANGE = np.arange(50, 350, 50).tolist()
# N2_RANGE = np.arange(1, 2, 1).tolist()
# N3_RANGE = np.arange(10, 15, 5).tolist()

# TRES MEDIAS
# N1_RANGE = np.arange(5, 35, 5).tolist() # NO SE PUEDEN PASAR DECIMALES VAYA MIERDA...
# N2_RANGE = np.arange(40, 80, 10).tolist() # PORQUE BACKTRADER NO LO ADMITE EN LOS INDICADORES...
# N3_RANGE = np.arange(50, 250, 50).tolist() # ULTIMO DATO NO SE ITERA

# PIVOT ZONE TEST
N1_RANGE = np.arange(2, 5, 1).tolist() # NO SE PUEDEN PASAR DECIMALES VAYA MIERDA... 
N2_RANGE = np.arange(50, 450, 50).tolist() # PORQUE BACKTRADER NO LO ADMITE EN LOS INDICADORES...
N3_RANGE = np.arange(4, 6, 1).tolist() # ULTIMO DATO NO SE ITERA

# ALEATORIO
# N1_RANGE = np.arange(14, 15, 1).tolist() # NO SE PUEDEN PASAR DECIMALES VAYA MIERDA... 
# N2_RANGE = np.arange(300, 310, 10).tolist() # PORQUE BACKTRADER NO LO ADMITE EN LOS INDICADORES...
# N3_RANGE = np.arange(4, 5, 1).tolist() # ULTIMO DATO NO SE ITERA

# ==================== DATOS ====================
DATA_FORMAT = 'tradingview'     # 'tradingview' | 'binance'
DATA_FOLDER_01 = 'data01'            # Carpeta con archivos CSV
#DATA_FOLDER_02 = 'data02'            # Carpeta con archivos CSV
#DATA_FOLDER_03 = 'data03'
#DATA_FOLDER_04 = 'data04'
#DATA_FOLDER_05 = 'data05'
#DATA_FOLDER_06 = 'data06'
#DATA_FOLDER_07 = 'data07'
# ... 
TIMEZONE = "Europe/Madrid"# Carpeta con archivos CSV
PERIODS_PER_YEAR = 525600 # Se utiliza para calcular las métricas de rendimiento, como Sharpe Ratio, Sortino Ratio, etc..
# PERIODS_PER_YEAR = 24*252       # 1h bolsa
# PERIODS_PER_YEAR = 390*252      # 1m bolsa USA (6.5h/día)
# PERIODS_PER_YEAR = 365*24       # 1h 24/7 (crypto/FX)
# PERIODS_PER_YEAR = 365*24*60    # 1m 24/7

# ==================== EJECUCIÓN ====================
# Eliminamos Bactesting = True porque ya de por si el programa lo hace.
# Eliminamos Optimize = True porque siempre estaremos optimizando, cuando solo haya un valor por parametro se pasaran rangos de un valor en la optimizacion.
OPTIMIZE_MAXIMIZE = 'Expectancy [%]'

# Métricas disponibles (puedes usar cualquiera de estas):
# - 'Equity Final [$]'          → Maximiza
# - 'Equity Peak [$]'           → Maximiza
# - 'Buy & Hold Return [%]'     → Maximiza
# - 'Return [%]'                → Maximiza
# - 'Return (Ann.) [%]'         → Maximiza
# - 'Volatility (Ann.) [%]'     → Minimiza (menor volatilidad = mejor)
# - 'Sharpe Ratio'              → Maximiza (retorno ajustado por riesgo)
# - 'Sortino Ratio'             → Maximiza (solo considera downside)
# - 'Calmar Ratio'              → Maximiza (retorno / drawdown)
# - 'Max. Drawdown [%]'         → Minimiza (menor pérdida máxima = mejor)
# - 'Avg. Drawdown [%]'         → Minimiza
# - 'Max. Drawdown Duration'    → Minimiza (menos tiempo en pérdida = mejor)
# - 'Avg. Drawdown Duration'    → Minimiza
# - 'Trades'                    → Maximiza (más operaciones)
# - 'Win Rate [%]'              → Maximiza (mayor % de aciertos)
# - 'Avg.Win'                    → Maximiza (ganancia media todos trades) (NUEVO)
# - 'Best Trade [%]'            → Maximiza
# - 'Avg Loss'                   → Minimiza (Perdida media de todos los trades) (NUEVO)
# - 'Worst Trade [%]'           → Minimiza (menos negativo = mejor)
# - 'Avg. Trade [%]'            → Maximiza
# - 'Max. Trade Duration'       → Depende de tu estrategia
# - 'Avg. Trade Duration'       → Depende de tu estrategia
# - 'Profit Factor'             → Maximiza (ganancias/pérdidas), > 1 = buena esperanza matematica
# - 'Expectancy [%]'            → Maximiza (beneficio esperado por trade) Métrica principal para largo plazo
# - 'SQN'                       → Maximiza (System Quality Number)

# OPTIMIZE_CONSTRAINT: Restricción para evitar combinaciones inválidas o redundantes Descomentar.
OPTIMIZE_CONSTRAINT = None
# OPTIMIZE_CONSTRAINT = 'n1 < n2'
# OPTIMIZE_CONSTRAINT = 'n2 < n3'
# OPTIMIZE_CONSTRAINT = 'n1 < n2 and n2 < n3'

# 4. SEPARACIÓN MÍNIMA (evitar medias muy cercanas):
# OPTIMIZE_CONSTRAINT = 'n1 < n2 and (n2 - n1) >= 10'
# OPTIMIZE_CONSTRAINT = 'n1 < n2 < n3 and (n2 - n1) >= 10 and (n3 - n2) >= 50'
#
# 5. RATIOS ESPECÍFICOS (n2 debe ser al menos 2x n1):
# OPTIMIZE_CONSTRAINT = 'n1 < n2 and n2 >= n1 * 2'
#
# 6. RANGOS CONDICIONALES (si n1 es pequeño, n2 debe ser grande):
# OPTIMIZE_CONSTRAINT = '(n1 < 20 and n2 > 50) or (n1 >= 20 and n2 >= 60)'
#
# 7. EVITAR VALORES ESPECÍFICOS (no usar 50 en n1 o n2):
# OPTIMIZE_CONSTRAINT = 'n1 != 50 and n2 != 50'
#
# 8. MÚLTIPLOS (n3 solo en múltiplos de 50: 50, 100, 150...):
# OPTIMIZE_CONSTRAINT = 'n1 < n2 and n3 % 50 == 0'
#
# 9. FUNCIONES MATEMÁTICAS (disponibles: abs, min, max, round, int, float):
# OPTIMIZE_CONSTRAINT = 'n1 < n2 and abs(n2 - n1) >= 15'
# OPTIMIZE_CONSTRAINT = 'min(n1, n2) >= 5 and max(n2, n3) <= 200'
#
# 10. COMBINACIONES COMPLEJAS (múltiples condiciones):
# OPTIMIZE_CONSTRAINT = 'n1 < n2 < n3 and (n2 - n1) >= 5 and n3 >= 100 and n1 != n2'

# ==================== BROKER (Capital y Costes) ====================
CASH = 20000   # Capital inicial en $

# Apalancamiento (formato intuitivo):
# MARGIN = 1  → Sin apalancamiento (1:1) - Necesitas 100% del valor
# MARGIN = 5  → Apalancamiento 1:5 - Puedes controlar 5x tu capital
# MARGIN = 10 → Apalancamiento 1:10 - Típico en CFDs
# MARGIN = 20 → Apalancamiento 1:20 - Típico en Forex
# MARGIN = 100 → Apalancamiento 1:100 - Muy arriesgado
MARGIN = 5         # Sin apalancamiento (recomendado)

# Comisiones (se aplican en entrada Y salida) 0.0025% Darwinex
COMMISSION = 0.0025 # Comisión por operación (%) Entrada y salida.

# Ejecución de órdenes
# True = Ejecuta al cierre de la barra (más realista, evita look-ahead bias)
# False = Ejecuta al open de la siguiente barra
TRADE_ON_CLOSE = True

# ==================== RESULTADOS ====================
# Resumen final
SHOW_RESULTS_SUMMARY = True
SUMMARY_FIRST_COL = 'Equity Final [$]'

# ==================== ANÁLISIS DE ROBUSTEZ ====================
# Activar/desactivar cada tipo de análisis
USE_CSCV = False                # CSCV + PBO (Combinatorial Symmetric Cross-Validation)
USE_STRESS = False              # Stress test de costes
USE_WF = False                  # Walk-Forward analysis

# --- Configuración CSCV (Cross-Validation) ---
# Número de particiones para CSCV
# Para 4 años de datos y evaluar cada trimestre: 4 años × 4 trimestres = 16
# IMPORTANTE: Cada bloque debe tener suficientes datos para calcular indicadores
CSCV_PARTITIONS = 16

# Top K candidatos a evaluar en CSCV
CSCV_TOP_K = 50

# Pool de candidatos (% del total ordenado por métrica)
# 0.30 = top 30% de las combinaciones
CSCV_TOP_POOL = 0.30

# Parámetros adicionales para CSCV
CSCV_N_SAMPLES = 1000  # Número de muestras para PBO
CSCV_N_JOBS = 1        # Procesamiento paralelo (1 = secuencial)

# --- Configuración Stress Test ---
# Grid de costes a probar (comisión y slippage)
STRESS_COST_GRID = {
    "commission": [0.0000, 0.0005, 0.0010],     # 0%, 0.05%, 0.1%
    "slippage_bps": [0, 5, 10]                   # 0, 5, 10 bps
}

# Rangos para stress test (compatibilidad)
STRESS_COMMISSION_RANGE = [0.0000, 0.0005, 0.0010]
STRESS_SLIPPAGE_RANGE = [0, 5, 10]

# --- Configuración Walk-Forward ---
WF_CONFIG = {
    "n_folds": 5,           # Número de ventanas temporales
    "scheme": "rolling"     # Esquema de ventanas (no usado aún)
}

# Parámetros adicionales para Walk-Forward
WF_N_SPLITS = 5      # Número de splits temporales
WF_TRAIN_SIZE = 0.7  # 70% para entrenamiento, 30% para test

# --- Configuración común para robustez ---
GRID_TOP_POOL = 0.30        # Pool de candidatos para stress/WF
GRID_MAX_CAND = 24          # Máximo de candidatos a evaluar


# ==================== EJECUTAR MAIN.PY ====================
# Esto permite ejecutar el programa haciendo: python config.py
if __name__ == "__main__":
    # Forzar cwd al directorio del proyecto para evitar mezclar rutas relativas
    # con otros repos cuando se lanza este config.py desde fuera.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 70)
    print("  Iniciando Backtest Bot v6.0 desde config.py")
    print("=" * 70)
    print(f"Configuración cargada:")
    print(f"  - Estrategias: {[s.__name__ for s in STRATEGIES]}")
    print(f"  - Capital: ${CASH:,}")
    print(f"  - Apalancamiento: {'Sin apalancamiento' if MARGIN == 0 else f'1:{MARGIN}'}")
    print(f"  - Comisión: {COMMISSION}%")
    print(f"  - CSCV: {'ACTIVADO' if USE_CSCV else 'DESACTIVADO'}")
    print("=" * 70)
    print()

    # Importar y ejecutar main.py
    import main
    main.main()
