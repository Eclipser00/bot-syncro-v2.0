# WF (Walk-Forward)

### ¿Qué es?

Validación fuera de muestra por ventanas temporales: se parte el histórico en folds (bloques), se re-simula cada candidato de parámetros en cada bloque OOS y se resume su consistencia.

### WF_median_sharpe_oos:

mediana del Sharpe en OOS (señal central de rendimiento robusto).

### WF_iqr_sharpe_oos:

IQR (p75−p25) del Sharpe OOS (volatilidad de resultados; más bajo = mejor).

(Opcionales) WF_mean/min/max_sharpe_oos, WF_pos_rate (% de folds con Sharpe>0), WF_return_oos_%.

### WF_folds:

nº de bloques OOS efectivamente evaluados (cuanto mayor, mejor comparabilidad).

## Cómo interpretar

0 ⇒ peor del grupo; 1 ⇒ mejor del grupo (según q5–q95, robusto).

WF_OOS_median_01: qué tan alto es el desempeño OOS (mediana) frente al resto.

WF_OOS_iqr_01: qué tan estable es (IQR bajo ⇒ valor alto).(En el codigo se modifica para una mejor lectura, valor alto para todas metricas.)

WF_OOS_score_01: mezcla simple de ambas para ranking rápido.

### Si hay empates,

usa como tie-breaker mayor <rank_metric> y, si está, WF_pos_rate más alto.

Evita candidatos con min OOS muy negativo o IQR enorme (picos no reproducibles).