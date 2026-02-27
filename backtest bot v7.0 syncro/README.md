# Flujo propuesto para v6.0 (con anti-overfitting y mesetas):

## Disclaimer

**Todo el código de este proyecto ha sido desarrollado utilizando vibe coding (codificación asistida por IA).**

Este proyecto es el resultado de desarrollo colaborativo con herramientas de inteligencia artificial. El código ha sido generado y refinado mediante asistencia de IA, siguiendo las mejores prácticas de programación y los estándares establecidos en el proyecto.

## Preparación inicial

Tú pones CSV en /data.

Configuras config.py (qué estrategias, optimización, parámetros, formato, tipo de chequeo de overfitting: CSCV, DSR, etc.).

## Carga de datos

Se leen los CSV con functions.csv_to_df_*.

Se validan con functions.validate_ohlcv().

Se guardan logs de: activo, nº de filas, rango temporal.

## Backtest base

Por cada activo y estrategia:

Se construye el grid de parámetros (build_param_grid).

Para cada combinación → strategies.run_strategy().

Se obtiene equity_curve, trades, métricas base.

Se guarda en /stats (cada combinación).

## Visualización preliminar

Se crean heatmaps de métricas (equity, DD, Sharpe).

Aquí ya puedes ver dónde están los “picos” y las “mesetas”.

## Anti-overfitting (overfitting.py)

Cada set de parámetros se pasa por:

CSCV + PBO → probabilidad de overfitting.

Stress de costes → sensibilidad a cambios en comisiones/slippage. solo OPTIMIZE = False, para un parametro concreto. (AUN SIN PROBAR FUNCIONAMIENTO)

Walk-Forward → estabilidad IS/OOS. (AUN SIN PROBAR FUNCIONAMIENTO)

## Selección de meseta robusta

Se filtra top X% de sets por equity/Sharpe OOS.

Se agrupan en clusters (vecindarios de parámetros).

Se elige un clúster estable (baja varianza, DD acotado, robustez alta).

Se selecciona un set representativo (mediana del clúster).

## Outputs finales

Solo el set robusto elegido se exporta a:

/plots: curva de equity y DD.

/trades: Excel con los trades.

/results: Excel con un resumen (1 fila por activo, con métricas + robustez).

Se loguea todo el proceso: desde parámetros hasta por qué se eligió ese set.

## 👉 En resumen:

Tu v4 buscaba el mejor resultado (maximize sobre un parámetro).

Tu v5 buscará una meseta robusta, donde los parámetros no sean frágiles y pasen filtros de sobreajuste.

Tu  v6 Cambio de libreria a Backtrader para mejorar estrategias.

![License: Personal Non-Commercial](https://img.shields.io/badge/license-personal_non_commercial-informational)

## Descargar CSV desde MT5 hacia data01

Script disponible:

- `download_mt5_data.py`

Objetivo:

- Descargar OHLCV desde MetaTrader5 para varios tickers.
- Exportar al formato exacto que ya usa `data01`:
  - `time,open,high,low,close,Volume`
  - `time` en epoch UNIX (segundos, UTC).
- Sobrescribir `data01/<TICKER>.csv`.

Como usar:

1. Edita los toggles al inicio del script:
   - `START_UTC = "YYYY-MM-DD HH:MM:SS"`
   - `END_UTC = "YYYY-MM-DD HH:MM:SS"`
   - `TICKERS = ["EURUSD", "GBPUSD", "USDJPY"]`
   - `TIMEFRAME = "M1"`
   - `OUTPUT_DIR = "data01"`
   - `OVERWRITE = True`
2. Ejecuta desde `backtest bot v7.0 syncro`:
   - `.\.venv\Scripts\python.exe .\download_mt5_data.py`

Notas importantes:

- Las fechas se interpretan siempre en UTC explicito.
- Si un simbolo no devuelve datos en ese rango, se reporta warning y no se crea CSV vacio.
- Si hay fallos parciales, el script sigue con los demas simbolos y muestra resumen.

## Changelog de logica

- Fecha: 2026-02-27
  - Cambios:
    - Se agrega `download_mt5_data.py` para descargar OHLCV desde MetaTrader5 con toggles en archivo.
    - Exporta CSV en formato `time,open,high,low,close,Volume` dentro de `data01`.
    - Valida `START_UTC/END_UTC`, timeframe y lista de simbolos; soporta fallos parciales con resumen.
  - Archivos:
    - download_mt5_data.py
    - README.md
  - Impacto esperado:
    - Facilita refrescar datos historicos MT5 para backtest sin depender de export manual.

- Fecha: 2026-02-22
  - Cambios:
    - `config.py` ahora fija el `cwd` al directorio del proyecto cuando se ejecuta con `python config.py`.
  - Archivos:
    - config.py
    - README.md
  - Impacto esperado:
    - Evita resolver rutas relativas contra la carpeta desde la que se lanza el comando.

- Fecha: 2026-02-20
  - Cambios:
    - `PivotZone` fija ATR en periodo 14 de forma interna para calcular `width`.
    - `PivotZoneTest` reutiliza `n1` como multiplicador de distancia minima entre zonas (`min_distance = new_width * n1`).
    - `parity_runner.py` resuelve CSVs de forma robusta cuando los nombres vienen con sufijos como `EURUSD,.csv`.
  - Archivos:
    - strategies.py
    - parity_runner.py
    - README.md
  - Impacto esperado:
    - El ancho ATR deja de depender de optimizacion.
    - `n1` pasa a controlar directamente el filtro de cercania entre zonas.
    - La paridad no falla por diferencias de naming en archivos CSV.

- Fecha: 2026-02-18
  - Cambios:
    - `PivotZoneTest` endurece el filtro de cercania entre zonas: `min_distance` pasa de `new_width * 2.0` a `new_width * 3.0` en los dos bloques de guardado de zonas.
  - Archivos:
    - strategies.py
    - README.md
  - Impacto esperado:
    - Menos zonas pivote aceptadas cuando quedan muy juntas en precio.

- Fecha: 2026-02-15
  - Cambios:
    - `Backtest.run_backtrader_core` activa auto-resample para `PivotZoneTest` cuando solo hay un feed valido en `df_master` (caso `DATA_FOLDER_01` con CSV M1).
    - Se crea internamente un segundo feed `data1` resampleado a `M3` para `TF_zone`, manteniendo `TF_entry` y `TF_stop` en `M1`.
    - `config.py` deja activos por defecto solo los datos de `DATA_FOLDER_01`; `DATA_FOLDER_02/03` quedan opcionales.
  - Archivos:
    - backtest.py
    - config.py
    - README.md
  - Impacto esperado:
    - `PivotZoneTest` puede ejecutarse con un unico set de CSV M1 (sin `DATA_FOLDER_02/03`) y conservar la logica multi-timeframe entry=M1, zone=M3.

- Fecha: 2026-02-12
  - Cambios:
    - `PivotZoneTest` ahora permite sobrescribir la ruta del JSONL de backtest con `BACKTEST_EVENTS_PATH`.
    - `tools/run_parity.py` exporta `BACKTEST_EVENTS_PATH=outputs/backtest_events.jsonl` para que el comparador lea el mismo archivo generado por el backtest.
  - Archivos:
    - strategies.py
    - ../tools/run_parity.py
    - README.md
  - Impacto esperado:
    - El reporte de paridad deja de mostrar `backtest events: 0` por desalineacion de rutas y vuelve a comparar eventos reales en ambos lados.

- Fecha: 2026-02-12
  - Cambios:
    - Se corrige la resolucion de rutas de salida para que siempre apunten al directorio del proyecto (`backtest bot v7.0 syncro`) y no al `cwd` desde donde se lanza Python.
    - `ensure_dirs` y `erase_directory` ahora operan con rutas absolutas ancladas a `data.py`.
    - Guardados de `grid/heatmaps/stats/trades/cscv/stress/wf/plots/results` usan rutas absolutas del proyecto.
    - `PivotZoneTest` corrige el path de `outputs/backtest_events.jsonl` para no escribir en la carpeta padre.
  - Archivos:
    - data.py
    - backtest.py
    - main.py
    - strategies.py
    - README.md
  - Impacto esperado:
    - El backtest deja de crear carpetas duplicadas en la raiz externa y reutiliza siempre las carpetas internas del proyecto.

- Fecha: 2026-01-31
  - Cambios:
    - PivotZone calcula ATR con TA-Lib para igualar la semantica del bot y zone_state siempre expone width aunque mid no exista (fallback).
    - Los eventos order_fill/position usan size en magnitud positiva y pueden fijar el bar_index del submit para paridad.
    - Sizing admite overrides por entorno (PARITY_EQUITY, PARITY_LEVERAGE, PARITY_PCT_CAP) para comparar con equity fija.
  - Archivos:
    - strategies.py
    - README.md
  - Impacto esperado:
    - Paridad mas estable entre backtest y bot en TF_zone, fills y sizing.

- Fecha: 2026-01-30
  - Cambios:
    - Se aÃ±ade instrumentaciÃ³n de paridad en la estrategia `PivotZoneTest` para emitir eventos JSONL `event_type=pivot_confirmed` cuando avanza `TF_zone`.
    - El runner `parity_runner.py` se usa para ejecutar `PivotZoneTest` sobre los mismos CSV del bot y producir `outputs/backtest_events.jsonl`.
  - Archivos:
    - strategies.py
    - parity_runner.py
    - README.md
  - Impacto esperado:
    - MÃ¡s trazabilidad para aislar divergencias entre backtest y bot en TF_zone (pivotes/zonas).


- Fecha: 2026-02-24
  - Cambios:
    - `PivotZoneTest` alinea entradas con live: mantiene origen `2/3` dentro + `2 cierres` fuera y elimina filtro por extremo/follow-through.
    - `stop inicial` pasa a ser el ultimo pivote confirmado de `TF_stop` dentro de la zona rota (LONG=min, SHORT=max).
    - Se elimina el trailing estructural en ejecucion para mantener salida solo por `SL inicial` o `TP` de la zona objetivo.
  - Archivos:
    - strategies.py
    - README.md
  - Impacto esperado:
    - Paridad de reglas con el bot live.
    - Menor complejidad de gestion de salida (sin trailing).
