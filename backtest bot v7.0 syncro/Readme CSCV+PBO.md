# 2. ¿Qué es CSCV? (Combinatorially Symmetric Cross Validation)

Es una técnica para evaluar estrategias o configuraciones de parámetros de forma combinatoria, usando todos los posibles cortes del histórico en train/test.

### Mecánica:

Divides tu histórico en N bloques de tiempo (ej. 8 meses → 8 bloques de 1 mes).

Tomas todas las combinaciones posibles:

Un subconjunto de bloques como train (optimizar).

El resto como test (validar).

Recorres todas las combinaciones (simétrico = cada bloque aparece igual de veces en train y test).

Para cada combinación mides la métrica (ej. Sharpe, MAR).

### 👉 Resultado:

Obtienes una distribución de rendimientos en test para cada set de parámetros.

Ves qué sets son consistentes en distintos cortes temporales.

✅ Ventaja: reduce el riesgo de “mejor set por pura suerte”.

## 🔹 3. ¿Qué es PBO? (Probability of Backtest Overfitting)

Es un unico valor global derivada de CSCV que mide la probabilidad de que tu se de parametros para optimizar esté sobreajustado.

Cómo se calcula:

Con CSCV ya tienes rendimiento en train y en test para cada set de parámetros.

### Comparas:

¿Los parámetros que son “top” en train también lo son en test?

¿O solo brillan en train pero fallan en test?

El PBO es la probabilidad de que un set que parece ganador en train acabe siendo perdedor en test.

### 👉 Interpretación:

PBO bajo (<20–30%) → tu estrategia es probablemente robusta.

PBO alto (>50%) → la estrategia está casi seguro sobreajustada (demasiada dependencia del histórico).

# ¿Qué devuelve la función?

PBO (global) te dice si el procedimiento de selección tiende a sobreajustar.

Datos para ajustar a un solo valor que mejor se comporta. CSCV_u_, CSCV_rank, win_rate 

### Interpretación aproximada PBO:

< 0.2 → buena señal (baja prob. de sobreajuste).

~ 0.5 → “cara o cruz” (neutro).

> 0.7 → mala pinta (probable sobreajuste).

### Interpretacion datos para "mejor valor":

Por-combo, filtra candidatos con CSCV_u_median bajo y CSCV_rank_median bajo (y poca diferencia entre mean y median).

Alto win_rate + buenos u/ranks OOS ⇒ candidato fuerte.

Alto win_rate + malos u/ranks OOS ⇒ sospecha de sobreajuste.

Bajo win_rate + buenos u/ranks OOS ⇒ estable aunque rara vez “parezca el nº1” en IS (suele ser material de meseta)

# main.py

## CSCV_PARTITIONS = 10

Qué controla: en cuántos bloques temporales se parte tu histórico. En cada fold, 1 bloque es OOS y el resto es IS (tipo “leave-one-block-out”).

Efecto en calidad vs. coste:

Más particiones ⇒ más comprobaciones OOS (diagnóstico de sobreajuste más fino), pero cada bloque es más corto y sube el coste de cómputo.

Menos particiones ⇒ menos coste, pero señal OOS más ruidosa.

Reglas rápidas (por nº de barras por activo):

Histórico pequeño (≤ ~2.000 barras): 5–8 particiones.

Medio (2.000–20.000): 8–12 particiones.

Grande (≥ ~20.000): 12–20 particiones.

Complejidad aproximada: por fold se re-simulan ~2 * top_k veces (IS + OOS). Total ≈ 2 * top_k * n_partitions backtests por activo y estrategia.

## CSCV_TOP_K = 10

Qué controla: cuántas combinaciones de parámetros (del grid) entran en el CSCV. Se seleccionan las top por la métrica elegida.

Efecto:

Subir top_k ⇒ mejor representatividad (menos riesgo de “ganador casual”), pero más re-simulaciones.

Bajar top_k ⇒ más rápido, pero la estimación de PBO puede volverse inestable.

Reglas rápidas (según el tamaño del grid):

Grid pequeño (≤ ~100 filas): 10–20.

Grid medio (100–500): 15–30.

Grid grande (≥ ~500): 20–50 o bien ~5–10% del grid (lo que sea mayor).

## “CSCV_METRIC” dinámico (cómo decide la métrica)

En tu main.py usamos un selector que elige automáticamente la métrica con la que:

rankeamos el grid,

elegimos el ganador IS, y

medimos el ranking OOS.

El orden de preferencia es:

'Sharpe Ratio' si existe en el grid,

la métrica que pusiste en OPTIMIZE_MAXIMIZE (p. ej. 'Equity Final [$]'),

'Return [%]' si existe,

si no, la última columna numérica disponible.

## Consejos

Si quieres que DSR también funcione (y que CSCV use Sharpe), optimiza por 'Sharpe Ratio' (OPTIMIZE_MAXIMIZE = 'Sharpe Ratio') para que esa columna aparezca en el grid.

Si prefieres evaluar por Equity (p. ej. en intradía con pocos trades), deja OPTIMIZE_MAXIMIZE = 'Equity Final [$]' y CSCV usará esa.

La métrica asumida es de “mayor es mejor”. Para métricas de “menor es mejor” (p. ej. Drawdown), no las uses como metric directamente salvo que inviertas el signo antes (nosotros, por defecto, no las elegimos como primera opción).

## Recetas rápidas

Setup conservador (rápido): CSCV_PARTITIONS=8, CSCV_TOP_K=10.

Setup balanceado (recomendado): CSCV_PARTITIONS=10–12, CSCV_TOP_K=15–25.

Setup estricto (más caro): CSCV_PARTITIONS=15–20, CSCV_TOP_K=30–50.

Regla mental: estima el coste ≈ 2 * top_k * partitions * (#activos) * (#estrategias) backtests. Ajusta para que te quepa en tiempo razonable.

# IS y OOS:

### IS (In-Sample) = “dentro de muestra”.

Es el trozo de histórico que usas para elegir/ajustar los parámetros (equivale a train). Ahí haces grid search, eliges el “mejor” por tu métrica, miras heatmaps, etc.
Clave: lo tocas para aprender; por eso está expuesto a sobreajuste.

### OOS (Out-of-Sample) = “fuera de muestra”.

Es el trozo que NO se usó para elegir y sirve solo para evaluar (equivale a test). Con los parámetros elegidos en IS, los fijas y re-simulas en OOS.
Clave: es tu foto “honesta” de cómo generaliza; no se toca para ajustar.

### En CSCV / Walk-Forward:

Partes el histórico en varios bloques (p. ej., 10).

En cada fold, tomas 1 bloque como OOS y el resto como IS (tipo leave-one-block-out).

Proceso por fold:

En IS seleccionas la mejor combinación (entrenas/eliges).

Con esos parámetros, evalúas en OOS.

Repites para todos los bloques. Si el “ganador IS” se hunde a menudo en OOS, huele a sobreajuste (PBO alto).