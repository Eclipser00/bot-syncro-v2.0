## Stress de costes (≤15 líneas):

### Qué es:

test de sensibilidad de la estrategia a los costes (comisión + slippage)

### Stress_Sensitivity (clave):

menor pendiente de Equity ~ cost_total; más bajo = mejor (poca caída al subir costes).

### Equity_at_min_cost y Equity_at_max_cost:

equity simulada en el menor y mayor coste del cost_grid.

Gap pequeño entre ambas ⇒ robustez.

Gap grande ⇒ la estrategia depende de costes ultra bajos.

Si empatan, prefiere el que tenga <rank_metric> más alto (mejor rendimiento base).
