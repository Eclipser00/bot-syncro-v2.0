# Bot de trading (estado actual del repo)

## 1) Resumen ejecutivo del proyecto
- Este repo implementa un bot de trading orientado a objetos que coordina descarga de datos OHLCV, evaluacion de riesgo, generacion de senales y ejecucion de ordenes mediante un broker. (ref: bot_trading/application/engine/bot_engine.py:TradingBot, bot_trading/infrastructure/data_fetcher.py:MarketDataService, bot_trading/application/engine/signals.py:Signal, bot_trading/application/engine/order_executor.py:OrderExecutor)
- El entrypoint principal es `bot_trading/main.py:main()`, se ejecuta como modulo Python y lee toda la configuracion desde `config.py`. (ref: bot_trading/main.py:main, config.py)
- El broker real soportado es MetaTrader5 via `MetaTrader5Client`, y hay un `FakeBroker` de ejemplo para ejecucion local. (ref: bot_trading/infrastructure/mt5_client.py:MetaTrader5Client, bot_trading/main.py:FakeBroker)
- En modo development, `FakeBroker` simula fills de SL/TP usando el OHLC mas reciente (via `process_price_tick`); para paridad se puede forzar cierre "solo en close" con `PARITY_CLOSE_ON_CLOSE=1`, diferir el cierre 1 vela con `PARITY_CLOSE_DELAY=1`, registrar el fill al precio de cierre con `PARITY_CLOSE_PRICE_ON_CLOSE=1`, y omitir la validacion de SL/TP con `PARITY_SKIP_SLTP_VALIDATION=1`. (ref: bot_trading/main.py:FakeBroker.process_price_tick, bot_trading/application/engine/bot_engine.py:TradingBot.run_once, bot_trading/application/strategies/pivot_zone_test_strategy.py:PivotZoneTestStrategy)
- La unica estrategia concreta incluida es `PivotZoneTestStrategy`, con logica multi-timeframe basada en zonas pivote y breakouts. (ref: bot_trading/application/strategies/pivot_zone_test_strategy.py:PivotZoneTestStrategy)
- Los tres entornos comparten la misma lista de simbolos/estrategias declarada en `config.py` (ver `DEFAULT_SYMBOLS` y `DEFAULT_STRATEGIES`); cada simbolo puede sobreescribir n1/n2/n3/size_pct/p para PivotZone sin duplicar configuracion por entorno. (ref: config.py:DEFAULT_SYMBOLS, config.py:DEFAULT_STRATEGIES)
- El modo development recorre el mismo pipeline que produccion; la unica diferencia es la fuente de datos (CSVs en `data_development`) y que se usa `FakeBroker` para no enviar ordenes reales. (ref: config.py:ENVIRONMENTS, bot_trading/main.py:_build_broker, bot_trading/infrastructure/data_fetcher.py:DevelopmentCsvDataProvider)

## 2) Quickstart (instalacion, variables de entorno, comandos de ejecucion)
- Version de Python indicada por el proyecto: 3.10.11 (ref: AGENTS.md:Seccion 1)
- Instalar dependencias del proyecto: `pip install -r requirements.txt`. (ref: requirements.txt)
- Ejecutar el bot principal: `python -m bot_trading.main` (lee config.py) o simplemente `python config.py` para lanzarlo desde la configuracion. (ref: bot_trading/main.py:main, config.py:__main__)
- Si arrancas con `python config.py` desde otra carpeta, `config.py` fija automaticamente el `cwd` al directorio del proyecto para evitar rutas relativas cruzadas.
- Alternar broker real vs simulado: cambia `ACTIVE_ENV` y/o `BrokerConfig.use_real_broker` en `config.py`. (ref: config.py:ACTIVE_ENV, config.py:BrokerConfig)
- Probar conectividad con MT5: `python tests/check_mt5_connection.py` (script manual, no pytest). (ref: tests/check_mt5_connection.py:main)
- Config de VS Code para ejecutar el modulo: `.vscode/launch.json`. (ref: .vscode/launch.json)
- Variables de entorno para credenciales MT5 se cargan si `BrokerConfig.load_env_credentials=True` en `config.py` y se usan al validar configuracion. (ref: config.py:_maybe_load_broker_env, bot_trading/main.py:main)
- Ejecutar snippets Python via MCP `python` (CPython local, sin Pyodide): `C:\PROGRA~1\PYTHON~1\python.exe .agent/mcp_python_local.py --cwd . --code "print('hola desde mcp')"` (ref: .agent/mcp_python_local.py, C:\Users\ricar\.codex\config.toml)

## 2.1) Visualizador M3 (Bokeh)
- Al ejecutar el bot normal (`python config.py` o `python -m bot_trading.main`) el visualizador se activa automaticamente por defecto (development y production).
- Script principal: `python -m bot_trading.visualizer`.
- Modo development (por defecto termina al agotar CSV): `python -m bot_trading.visualizer --mode development --refresh-seconds 900`.
- Modo production (loop continuo): `python -m bot_trading.visualizer --mode production --refresh-seconds 900`.
- Salida HTML del runner CLI: `outputs/visualizer_m3.html` (ajustable con `--output-html`).
- Salida HTML del bot normal (`python config.py` / `python -m bot_trading.main`): un archivo por ticker en `outputs/visualizer{SYMBOL}.html`.
- `bot_events` se autodetecta en estas dos rutas y usa la mas reciente: `outputs/bot_events.jsonl` y `../outputs/bot_events.jsonl`.
- El visualizador inicia la lectura incremental de logs desde el final (tail) para evitar mezclar historico viejo de ejecuciones anteriores.
- Los eventos de entrada/fill/posicion se filtran por ventana temporal visible y por rango de precios de las velas del simbolo para evitar distorsiones del eje (outliers o precios `0`).
- Archivos generados por ticker:
  - `outputs/visualizerEURUSD.html`
  - `outputs/visualizerGBPUSD.html`
  - `outputs/visualizerUSDJPY.html`
- Variables opcionales:
  - `VISUALIZER_AUTO_ENABLE=0|1` (forzar desactivar/activar autovisualizador).
  - `VISUALIZER_REFRESH_SECONDS=<n>` (intervalo de refresco en segundos; default `900`).
- El visualizador mantiene buffers incrementales por simbolo:
  - velas `M3`,
  - zonas pivote desde `logs/pivot_zones.log`,
  - entradas/fills/posicion desde `outputs/bot_events.jsonl`, proyectadas al eje temporal `M3`.
- Render directo del backtest:
  - `bot_trading/visualization/plot_bokeh.py` carga `backtest bot v7.0 syncro/plotting.py`.
  - El dibujo principal se hace con `Plot.max_min_plot(...)` para mantener la misma estetica que el backtest bot.
  - El visualizador reconstruye `trades_df` desde eventos `order_fill` (open/close) para activar el trazado de lineas de posicion (entry->exit) que ya implementa `plotting.py`.
- Apariencia del plot:
  - velas OHLC del plotting del backtest,
  - lineas de zonas (`Zone_i_Top` / `Zone_i_Bot`),
  - marcadores de eventos (`Eventos_signal`, `Eventos_order_fill`, `Eventos_position`),
  - lineas de seguimiento de posiciones (`Trades`) cuando hay fills de apertura+cierre,
  - crosshair, hover y leyenda hide/show del mismo renderer base,
  - auto-fit de ancho al viewport del navegador para evitar desborde horizontal en pantallas pequenas.
- Regla robusta de escritura (Windows lock):
  - el guardado usa `temp + replace` atomico (`safe_write_html`),
  - si el HTML esta abierto y bloqueado (`PermissionError`/`OSError`), se registra el error y el proceso continua sin caerse.
- Limpieza de warnings:
  - el adaptador del visualizador filtra dos warnings benignos de Bokeh (`toolbar.active_drag` y `toolbar.active_scroll`) emitidos por `gridplot(..., merge_tools=True)` en `plotting.py` heredado.
- Modulos:
  - `bot_trading/visualization/data_source.py`
  - `bot_trading/visualization/event_reader.py`
  - `bot_trading/visualization/state_store.py`
  - `bot_trading/visualization/plot_bokeh.py`
  - `bot_trading/visualization/runner.py`

## 3) Arquitectura del repo (carpetas y responsabilidades)
- `bot_trading/domain`: entidades y dataclasses de dominio (ordenes, posiciones, limites de riesgo). (ref: bot_trading/domain/entities.py)
- `bot_trading/application`: logica de negocio del bot, motor, risk management, registry, estrategias. (ref: bot_trading/application/engine/bot_engine.py:TradingBot, bot_trading/application/risk_management.py:RiskManager, bot_trading/application/strategy_registry.py:StrategyRegistry, bot_trading/application/strategies/base.py:Strategy)
- `bot_trading/infrastructure`: integraciones externas (MT5, datos, exportacion). (ref: bot_trading/infrastructure/mt5_client.py:MetaTrader5Client, bot_trading/infrastructure/data_fetcher.py:MarketDataService, bot_trading/infrastructure/file_exporter.py:ExcelExporter)
- `tests`: suite de pruebas con pytest y scripts de verificacion manual. (ref: tests/*)
- `data_development`: CSVs historicos de ejemplo para el proveedor de datos de desarrollo. (ref: data_development/*.csv, bot_trading/infrastructure/data_fetcher.py:DevelopmentCsvDataProvider)
- `config.py`: configuracion declarativa por entornos, fuente de verdad consumida por el entrypoint principal. (ref: config.py, bot_trading/main.py:main)

## 4) Flujo de ejecucion end-to-end

### 4.1 Arranque
- El entrypoint `bot_trading/main.py:main()` configura logging y decide si usar MT5 real o `FakeBroker` segun `settings.broker.use_real_broker` en `config.py`. (ref: bot_trading/main.py:main, config.py:BrokerConfig)
- Si `use_real_broker=True`, se crea `MetaTrader5Client` con los retries definidos y se llama a `connect()`. (ref: bot_trading/main.py:main, bot_trading/infrastructure/mt5_client.py:MetaTrader5Client.connect)
- Si `use_real_broker=False`, se crea `FakeBroker` y se llama a `connect()` (simulado). (ref: bot_trading/main.py:FakeBroker.connect)

### 4.2 Carga de configuracion (fuente de verdad actual)
- Toda la configuracion activa se define en `config.py` (`settings` + `validate_config`), incluyendo broker, logging, riesgo, simbolos, estrategias, loop y datos. (ref: config.py:settings, config.py:validate_config)
- `bot_trading/main.py` importa y ejecuta `validate_config(settings)` en cada arranque; no hay constantes duplicadas en el entrypoint. (ref: bot_trading/main.py:main)
- `config.py` puede ejecutarse directamente (`python config.py`) para lanzar el bot sin abrir otros archivos. (ref: config.py:__main__)

### 4.3 Inicializacion de componentes
- Se crea `MarketDataService` con el broker para obtener datos de mercado. (ref: bot_trading/main.py:main, bot_trading/infrastructure/data_fetcher.py:MarketDataService)
- Se crea `RiskManager` con `RiskLimits` definidos en `config.py` (drawdown global, por activo y por estrategia). (ref: bot_trading/main.py:main, config.py:RiskConfig, bot_trading/application/risk_management.py:RiskManager)
- Se crea `OrderExecutor` con el broker. (ref: bot_trading/main.py:main, bot_trading/application/engine/order_executor.py:OrderExecutor)
- Se instancian estrategias `PivotZoneTestStrategy` y se definen simbolos `SymbolConfig` segun lo configurado en `config.py`. (ref: bot_trading/main.py:main, bot_trading/application/strategies/pivot_zone_test_strategy.py:PivotZoneTestStrategy, bot_trading/domain/entities.py:SymbolConfig, config.py:StrategyConfig)
- Se construye el `TradingBot` con todos los componentes. (ref: bot_trading/main.py:main, bot_trading/application/engine/bot_engine.py:TradingBot)

### 4.4 Bucle principal o pipeline
- El entrypoint ejecuta `TradingBot.run_synchronized(timeframe_minutes=1, wait_after_close=5, skip_sleep_when_simulated=settings.loop.skip_sleep_when_simulated)`. En development el flag viene en `True` (sin sleeps con CSV), en produccion sigue en `False`; puedes desactivarlo en dev si quieres la misma cadencia que live. (ref: bot_trading/main.py:main, config.py:LoopConfig, bot_trading/application/engine/bot_engine.py:TradingBot.run_synchronized)
- `skip_sleep_when_simulated` controla si el loop usa los timestamps del CSV sin dormir entre ciclos (acelerado) o se sincroniza con reloj real. (ref: bot_trading/application/engine/bot_engine.py:TradingBot.run_synchronized, bot_trading/infrastructure/data_fetcher.py:DevelopmentCsvDataProvider.get_simulated_now)
- En modo real el proveedor de datos es el broker (MT5) y en development es `DevelopmentCsvDataProvider`; el resto del flujo (riesgo, estrategias, ejecucion de ordenes) es identico. (ref: bot_trading/main.py:_build_market_data_service, bot_trading/infrastructure/data_fetcher.py:ProductionDataProvider)

### 4.5 Generacion de senales y decisiones
- `TradingBot.run_once()` sincroniza posiciones abiertas, actualiza historial de trades cerrados y valida limites de riesgo global/por simbolo/por estrategia. (ref: bot_trading/application/engine/bot_engine.py:TradingBot.run_once, bot_trading/application/risk_management.py:RiskManager)
- El bot calcula timeframes requeridos por estrategia y simbolo usando `strategy.timeframes` y `strategy.allowed_symbols` si existe. (ref: bot_trading/application/engine/bot_engine.py:TradingBot.run_once)
- El bot solicita datos al `MarketDataService` y entrega `data_by_timeframe` a cada estrategia para generar `Signal`. (ref: bot_trading/application/engine/bot_engine.py:TradingBot.run_once, bot_trading/infrastructure/data_fetcher.py:MarketDataService.get_data, bot_trading/application/engine/signals.py:Signal)
- La estrategia `PivotZoneTestStrategy` calcula zonas pivote y breakouts, genera senales BUY/SELL con SL/TP si hay condiciones, y no genera senales si no puede calcular size. (ref: bot_trading/application/strategies/pivot_zone_test_strategy.py:PivotZoneTestStrategy.generate_signals, bot_trading/application/strategies/pivot_zone_test_strategy.py:_calc_lot_size_by_stop)

### 4.6 Ejecucion / IO (broker, filesystem, red)
- `OrderExecutor.execute_order()` envia ordenes via `broker_client.send_market_order()` y actualiza un registro local de posiciones. (ref: bot_trading/application/engine/order_executor.py:OrderExecutor.execute_order)
- Si el broker es MT5, `MetaTrader5Client.send_market_order()` usa `mt5.order_send()`, valida volumen/tipo y normaliza `SL/TP` de mercado segun `trade_stops_level` y `trade_freeze_level` para reducir rechazos `10016 Invalid stops`. (ref: bot_trading/infrastructure/mt5_client.py:MetaTrader5Client.send_market_order)
- `PivotZoneTestStrategy` puede crear o modificar ordenes pendientes TP/SL (bracket) usando `broker_client.create_pending_order`, `cancel_order` y `modify_order` si existen. (ref: bot_trading/application/strategies/pivot_zone_test_strategy.py:_ensure_bracket_orders, bot_trading/infrastructure/mt5_client.py:MetaTrader5Client.create_pending_order)
- El exportador `ExcelExporter` puede escribir trades a un archivo Excel si se usa manualmente. (ref: bot_trading/infrastructure/file_exporter.py:ExcelExporter.export_trades)

### 4.7 Cierre y cleanup
- `main()` captura `KeyboardInterrupt` y registra estadisticas finales consultando el broker. (ref: bot_trading/main.py:main)
- `MetaTrader5Client.__del__()` intenta cerrar la conexion con `mt5.shutdown()` al destruir el cliente. (ref: bot_trading/infrastructure/mt5_client.py:MetaTrader5Client.__del__)

## 5) Componentes principales (por modulo/clase)

### 5.1 `bot_trading/main.py`
- **Que hace**: entrypoint, construye y ejecuta el bot con broker real o simulado. (ref: bot_trading/main.py:main)
- **Entradas**: configuracion declarativa desde `config.py` (broker/logging/riesgo/estrategias/simbolos/loop/datos). (ref: bot_trading/main.py:main, config.py:BotConfig)
- **Salidas**: logs a consola (y archivo si se configura) y ejecucion del bucle sincronizado. (ref: bot_trading/main.py:main)
- **Errores comunes**: fallo de conexion MT5 en `MetaTrader5Client.connect()` si el terminal no esta disponible. (ref: bot_trading/infrastructure/mt5_client.py:MetaTrader5Client.connect)
- **Notas**: `FakeBroker` define un flujo simulado, genera OHLCV lineal por minuto y ahora cierra posiciones cuando el close cruza SL/TP usando `process_price_tick` (invocado por `TradingBot.run_once`). (ref: bot_trading/main.py:FakeBroker.get_ohlcv, bot_trading/main.py:FakeBroker.process_price_tick, bot_trading/application/engine/bot_engine.py:TradingBot.run_once)

### 5.2 `TradingBot` (motor)
- **Que hace**: orquesta datos, riesgo, estrategias y ejecucion de ordenes. (ref: bot_trading/application/engine/bot_engine.py:TradingBot)
- **Entradas**: broker_client, MarketDataService, RiskManager, OrderExecutor, lista de estrategias y simbolos. (ref: bot_trading/application/engine/bot_engine.py:TradingBot)
- **Salidas**: ordenes enviadas al broker y actualizacion de `trade_history`. (ref: bot_trading/application/engine/bot_engine.py:TradingBot.run_once)
- **Errores comunes**: excepciones al obtener datos o por `StopIteration` en proveedores CSV. (ref: bot_trading/application/engine/bot_engine.py:TradingBot.run_once, bot_trading/infrastructure/data_fetcher.py:DevelopmentCsvDataProvider.get_data)

### 5.3 `OrderExecutor`
- **Que hace**: envia ordenes y mantiene un registro local de posiciones abiertas. (ref: bot_trading/application/engine/order_executor.py:OrderExecutor)
- **Entradas**: `OrderRequest` y broker con `send_market_order()`. (ref: bot_trading/domain/entities.py:OrderRequest, bot_trading/application/engine/order_executor.py:OrderExecutor.execute_order)
- **Salidas**: `OrderResult` y actualizacion de `open_positions`. (ref: bot_trading/application/engine/order_executor.py:OrderExecutor.execute_order)
- **Errores comunes**: fallos de broker al enviar ordenes (se reportan en logs). (ref: bot_trading/application/engine/order_executor.py:OrderExecutor.execute_order)

### 5.4 `Signal` y `SignalType`
- **Que hace**: contrato de salida de estrategias para el motor. (ref: bot_trading/application/engine/signals.py:Signal, bot_trading/application/engine/signals.py:SignalType)
- **Entradas/Salidas**: objetos con symbol, timeframe, tipo, size, SL, TP. (ref: bot_trading/application/engine/signals.py:Signal)

### 5.5 `RiskManager`
- **Que hace**: calcula drawdown y valida limites globales, por simbolo, por estrategia y por margen. (ref: bot_trading/application/risk_management.py:RiskManager)
- **Entradas**: lista de `TradeRecord` y `RiskLimits`. (ref: bot_trading/domain/entities.py:TradeRecord, bot_trading/domain/entities.py:RiskLimits)
- **Salidas**: booleanos para permitir o bloquear operaciones. (ref: bot_trading/application/risk_management.py:RiskManager.check_bot_risk_limits)
- **Errores comunes**: `AccountInfo.equity <= 0` bloquea ordenes por seguridad. (ref: bot_trading/application/risk_management.py:RiskManager.check_margin_limits)

### 5.6 `StrategyRegistry`
- **Que hace**: asigna Magic Numbers deterministas a estrategias por hash MD5. (ref: bot_trading/application/strategy_registry.py:StrategyRegistry.register_strategy)
- **Entradas/Salidas**: `strategy_name` -> `magic_number`. (ref: bot_trading/application/strategy_registry.py:StrategyRegistry.register_strategy)

### 5.7 `Strategy` (protocol)
- **Que hace**: define interfaz minima de estrategias (`name`, `timeframes`, `generate_signals`). (ref: bot_trading/application/strategies/base.py:Strategy)

### 5.8 `PivotZoneTestStrategy`
- **Que hace**: calcula zonas pivote en TF superior, detecta breakouts en TF de entrada y gestiona SL/TP con pivotes confirmados. (ref: bot_trading/application/strategies/pivot_zone_test_strategy.py:PivotZoneTestStrategy)
- **Entradas**: `data_by_timeframe` con OHLCV y `df.attrs["symbol"]`. (ref: bot_trading/application/strategies/pivot_zone_test_strategy.py:PivotZoneTestStrategy.generate_signals)
- **Salidas**: lista de `Signal` BUY/SELL con SL/TP y size calculado. (ref: bot_trading/application/strategies/pivot_zone_test_strategy.py:PivotZoneTestStrategy.generate_signals)
- **Errores comunes**: si no hay `broker_client._get_symbol_info`, el size no se calcula y no hay senales. (ref: bot_trading/application/strategies/pivot_zone_test_strategy.py:_calc_lot_size_by_stop)
- **Notas**: la construcción de zonas replica el indicador PivotZone del backtest (ATR + pivotes 3 velas, lock por cambio de rol) y guarda la zona solo en transición valid 0→1. El detector Pivot3Candle confirma el pivote con 1 vela de retraso (usa las 3 velas previas y confirma en la vela central). (ref: bot_trading/application/strategies/pivot_zone_test_strategy.py:_process_zone_tf)

### 5.9 `MarketDataService` y proveedores de datos
- **Que hace**: orquesta la obtencion y resample de OHLCV segun timeframes. (ref: bot_trading/infrastructure/data_fetcher.py:MarketDataService)
- **Produccion**: `ProductionDataProvider` usa `broker_client.get_ohlcv()` y cachea datos base. (ref: bot_trading/infrastructure/data_fetcher.py:ProductionDataProvider.get_data)
- **Desarrollo**: `DevelopmentCsvDataProvider` lee CSV y avanza un bar por llamada. (ref: bot_trading/infrastructure/data_fetcher.py:DevelopmentCsvDataProvider.get_data)

### 5.10 `MetaTrader5Client`
- **Que hace**: integra con la libreria MetaTrader5 para datos y ordenes. (ref: bot_trading/infrastructure/mt5_client.py:MetaTrader5Client)
- **Entradas**: simbolos/timeframes/ordenes. (ref: bot_trading/infrastructure/mt5_client.py:MetaTrader5Client.get_ohlcv, bot_trading/infrastructure/mt5_client.py:MetaTrader5Client.send_market_order)
- **Salidas**: DataFrames OHLCV, listas de posiciones/trades y `OrderResult`. (ref: bot_trading/infrastructure/mt5_client.py:MetaTrader5Client.get_open_positions, bot_trading/infrastructure/mt5_client.py:MetaTrader5Client.get_closed_trades)
- **Errores comunes**: `MT5ConnectionError` si `mt5.initialize()` falla; `MT5DataError` si el simbolo no existe; rechazos `10016 Invalid stops` cuando `SL/TP` quedan demasiado cerca o del lado incorrecto (el cliente ahora ajusta niveles al minimo valido). (ref: bot_trading/infrastructure/mt5_client.py:MT5ConnectionError, bot_trading/infrastructure/mt5_client.py:MT5DataError, bot_trading/infrastructure/mt5_client.py:MetaTrader5Client._normalize_market_stops)

### 5.11 `ExcelExporter`
- **Que hace**: exporta `TradeRecord` a Excel si se invoca manualmente. (ref: bot_trading/infrastructure/file_exporter.py:ExcelExporter.export_trades)

## 6) Configuracion (detallada)

### 6.1 Fuentes de configuracion y precedencia actual
- Fuente activa y unica: `config.py`, con `settings` y `validate_config()`; todo el entrypoint lee de aqui. (ref: config.py:settings, config.py:validate_config, bot_trading/main.py:main)
- Precedencia efectiva: `python -m bot_trading.main` y `python config.py` usan la misma configuracion declarativa. (ref: bot_trading/main.py:main, config.py:__main__)

### 6.2 Configuracion aplicada en `bot_trading/main.py`
- `BrokerConfig.use_real_broker` decide `MetaTrader5Client` (production) vs `FakeBroker` (development/testing). (ref: config.py:BrokerConfig, bot_trading/main.py:_build_broker)
 - `LoggingConfig` se aplica con `logging.basicConfig` (consola y opcional archivo) y además define la ruta del log dedicado de zonas pivote (`pivot_log_file_path`). (ref: config.py:LoggingConfig, bot_trading/main.py:_configure_logging)
- `RiskConfig` se mapea a `RiskLimits` para `RiskManager`. (ref: config.py:RiskConfig, bot_trading/application/risk_management.py:RiskManager)
- `StrategyConfig` y `SymbolConfigEntry` se usan para instanciar `PivotZoneTestStrategy` y `SymbolConfig`. (ref: config.py:StrategyConfig, bot_trading/main.py:_build_strategies)
- `LoopConfig` controla `run_synchronized(timeframe_minutes, wait_after_close, skip_sleep_when_simulated)`. (ref: config.py:LoopConfig, bot_trading/main.py:main, bot_trading/application/engine/bot_engine.py:TradingBot.run_synchronized)

### 6.3 Configuracion declarativa en `config.py`
- `ACTIVE_ENV` por defecto es "development" y controla `settings`. (ref: config.py:ACTIVE_ENV, config.py:settings)
- `BrokerConfig` campos: `use_real_broker`, `max_retries`, `retry_delay`, `load_env_credentials`, `env_prefix`, `server`, `login`, `password`. (ref: config.py:BrokerConfig)
- `LoggingConfig` campos: `level`, `log_to_file`, `log_file_path`, `pivot_log_file_path`, `format`. (ref: config.py:LoggingConfig)
- `RiskConfig` campos: `dd_global`, `dd_por_activo`, `dd_por_estrategia`, `initial_balance`, `max_margin_usage_percent`. (ref: config.py:RiskConfig)
- `SymbolConfigEntry` campos: `name`, `min_timeframe`, y overrides opcionales por símbolo para PivotZone (`n1`, `n2`, `n3`, `size_pct`, `p`). (ref: config.py:SymbolConfigEntry)
- `StrategyConfig` campos: `name`, `allowed_symbols` (si se deja vacío se autocompleta con todos los símbolos declarados), `tf_entry`, `tf_zone`, `tf_stop`, `n1`, `n2`, `n3`, `size_pct`, `p`, `timeframes`. (ref: config.py:StrategyConfig)
- `LoopConfig` campos: `timeframe_minutes`, `wait_after_close`, `skip_sleep_when_simulated`. (ref: config.py:LoopConfig)
- `DataConfig` campos: `data_mode`, `data_development_dir`, `bootstrap_lookback_days_zone`, `bootstrap_lookback_days_entry`, `bootstrap_lookback_days_stop`, `csv_base_timeframe`. (ref: config.py:DataConfig)

#### 6.3.1 Valores por entorno definidos en `ENVIRONMENTS`
- Todos los entornos reutilizan `DEFAULT_SYMBOLS` y `DEFAULT_STRATEGIES`; la diferencia radica en broker/datos/logging/riesgo. Cada símbolo puede definir overrides (n1/n2/n3/size_pct/p) y la estrategia completa `allowed_symbols` automáticamente con la lista de símbolos configurados si se deja vacía.
- `production`:
  - broker: `use_real_broker=True`, `max_retries=3`, `retry_delay=1.0`, `load_env_credentials=True`. (ref: config.py:ENVIRONMENTS)
  - logging: `level="INFO"`, `log_to_file=True`, `log_file_path="logs/production.log"`, `pivot_log_file_path="logs/pivot_zones.log"`. (ref: config.py:ENVIRONMENTS)
  - risk: `dd_global=30.0`, `dd_por_activo` EURUSD/GBPUSD/USDJPY=30.0, `dd_por_estrategia` PivotZoneTest=30.0, `initial_balance=20000.0` (default), `max_margin_usage_percent=80.0` (default). (ref: config.py:RiskConfig, config.py:ENVIRONMENTS)
  - symbols/strategies: se cargan desde `DEFAULT_SYMBOLS`/`DEFAULT_STRATEGIES` (actualmente EURUSD/GBPUSD/USDJPY en M1 con overrides por símbolo, y PivotZoneTest con M1/M3). (ref: config.py:DEFAULT_SYMBOLS, config.py:DEFAULT_STRATEGIES)
  - loop: `timeframe_minutes=1`, `wait_after_close=5`, `skip_sleep_when_simulated=False`. (ref: config.py:ENVIRONMENTS)
  - data: `data_mode="production"`, `data_development_dir="data_development"`, `bootstrap_lookback_days_zone=14`, `bootstrap_lookback_days_entry=None`, `bootstrap_lookback_days_stop=None`, `csv_base_timeframe="M1"`. (ref: config.py:ENVIRONMENTS)
- `development`:
  - broker: `use_real_broker=False`, `max_retries=1`, `retry_delay=0.1`, `load_env_credentials=False`. (ref: config.py:ENVIRONMENTS)
  - logging: `level="DEBUG"`, `log_to_file=True`, `log_file_path="logs/development.log"`, `pivot_log_file_path="logs/pivot_zones.log"`. (ref: config.py:ENVIRONMENTS)
  - risk: mismos limites que production para activos y estrategia PivotZoneTest. (ref: config.py:ENVIRONMENTS)
  - symbols/strategies: `DEFAULT_SYMBOLS`/`DEFAULT_STRATEGIES` (iguales a producción). (ref: config.py:DEFAULT_SYMBOLS, config.py:DEFAULT_STRATEGIES)
  - loop: `timeframe_minutes=1`, `wait_after_close=5`, `skip_sleep_when_simulated=True` por defecto (dev avanza sin sleeps; puedes ponerlo en False si quieres reloj real). (ref: config.py:ENVIRONMENTS)
  - data: `data_mode="development"`, `data_development_dir="data_development"`, `bootstrap_lookback_days_zone=14`, `bootstrap_lookback_days_entry=None`, `bootstrap_lookback_days_stop=None`, `csv_base_timeframe="M1"`. (ref: config.py:ENVIRONMENTS)
- `testing`:
  - broker: `use_real_broker=False`, `max_retries=1`, `retry_delay=0.0`, `load_env_credentials=False`. (ref: config.py:ENVIRONMENTS)
  - logging: `level="DEBUG"`, `log_to_file=False`, `pivot_log_file_path="logs/pivot_zones.log"`. (ref: config.py:ENVIRONMENTS)
  - risk: `dd_global=100.0` y sin limites por activo/estrategia. (ref: config.py:ENVIRONMENTS)
  - symbols/strategies: `DEFAULT_SYMBOLS`/`DEFAULT_STRATEGIES` (iguales a producción; permite probar múltiples símbolos/overrides en tests). (ref: config.py:DEFAULT_SYMBOLS, config.py:DEFAULT_STRATEGIES)
  - loop: `timeframe_minutes=1`, `wait_after_close=0`, `skip_sleep_when_simulated=True`. (ref: config.py:ENVIRONMENTS)
  - data: `data_mode="development"`, `data_development_dir="data_development"`, `bootstrap_lookback_days_zone=14`, `bootstrap_lookback_days_entry=None`, `bootstrap_lookback_days_stop=None`, `csv_base_timeframe="M1"`. (ref: config.py:ENVIRONMENTS)

#### 6.3.2 Validaciones y efectos en `validate_config()`
- Verifica que `ACTIVE_ENV` exista y que haya al menos un simbolo. (ref: config.py:validate_config)
- `tf_stop` se define siempre desde `config.py` y no se fuerza a `tf_zone`. (ref: config.py:validate_config)
- Normaliza lookbacks si `bootstrap_lookback_days_entry/stop` son `None`. (ref: config.py:validate_config)
- Obliga coherencia `broker.use_real_broker` vs `data.data_mode`. (ref: config.py:validate_config)
- Lanza `ValueError` si hay errores de configuracion. (ref: config.py:validate_config)

## 7) Datos

### 7.1 Formato OHLCV y estructuras en memoria
- Los DataFrames deben tener columnas `open`, `high`, `low`, `close`, `volume` y un `DatetimeIndex`. (ref: bot_trading/infrastructure/mt5_client.py:MetaTrader5Client.get_ohlcv, bot_trading/infrastructure/data_fetcher.py:_resample)
- El proveedor agrega `df.attrs["symbol"]` con el nombre del simbolo. (ref: bot_trading/infrastructure/data_fetcher.py:ProductionDataProvider.get_data, bot_trading/infrastructure/data_fetcher.py:DevelopmentCsvDataProvider.get_data)
- `PivotZoneTestStrategy` lee el simbolo desde `df.attrs["symbol"]`. (ref: bot_trading/application/strategies/pivot_zone_test_strategy.py:_get_symbol)
- El resample local etiqueta las velas en el cierre del intervalo (label/closed right), por lo que el indice del dataframe resampleado representa el cierre del bloque. (ref: bot_trading/infrastructure/data_fetcher.py:_resample)

### 7.2 CSV de desarrollo
- El CSV debe contener columnas `open`, `high`, `low`, `close` y `time` o `datetime` (la columna `volume` es opcional). (ref: bot_trading/infrastructure/data_fetcher.py:_load_csv)
- `time` se interpreta como Unix epoch en segundos o milisegundos; `datetime` se parsea como UTC. (ref: bot_trading/infrastructure/data_fetcher.py:_load_csv)
- Los archivos deben llamarse `{SYMBOL}.csv` en el directorio `data_development/` o el `data_dir` configurado. (ref: bot_trading/infrastructure/data_fetcher.py:_load_csv)

### 7.3 Timeframes soportados para resample
- Resample local soporta `M1`, `M3`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1`. (ref: bot_trading/infrastructure/data_fetcher.py:_TIMEFRAME_MAP)
- El cliente MT5 soporta tambien `W1` y `MN1` para descarga directa. (ref: bot_trading/infrastructure/mt5_client.py:TIMEFRAME_MAP)

### 7.4 Timezones
- `MetaTrader5Client.get_ohlcv()` normaliza `start/end` a UTC-aware y devuelve `DatetimeIndex` en UTC (`utc=True`). (ref: bot_trading/infrastructure/mt5_client.py:MetaTrader5Client.get_ohlcv)
- `ProductionDataProvider` normaliza `now`, cache e incremental_start a UTC para evitar errores por mezcla naive/aware. (ref: bot_trading/infrastructure/data_fetcher.py:ProductionDataProvider.get_data)
- `DevelopmentCsvDataProvider` usa timestamps con zona horaria UTC. (ref: bot_trading/infrastructure/data_fetcher.py:_load_csv)

## 8) Logging / Observabilidad
- El entrypoint aplica `LoggingConfig` de `config.py` (nivel, formato, consola y opcional archivo). (ref: config.py:LoggingConfig, bot_trading/main.py:_configure_logging)
- Modulos internos usan `logging.getLogger(__name__)` y emiten logs en `DEBUG/INFO/WARNING/ERROR`. (ref: bot_trading/application/engine/bot_engine.py, bot_trading/application/risk_management.py, bot_trading/infrastructure/mt5_client.py)
- Las zonas pivote consolidadas se escriben además en el archivo configurado en `LoggingConfig.pivot_log_file_path` (por defecto `logs/pivot_zones.log`) con símbolo, timeframe de zona y top/bot/mid/width en precisión de 8 decimales para evitar pérdida de resolución en forex; el timestamp registrado es el de la vela de datos (backtest/stream) y no la hora de ejecución. `_configure_logging` trunca tanto `log_file_path` (si se usa) como `pivot_log_file_path` al arrancar para contener sólo la ejecución actual. (ref: bot_trading/main.py:_configure_logging, bot_trading/application/strategies/pivot_zone_test_strategy.py:_process_zone_tf, config.py:LoggingConfig)

## 9) Errores y troubleshooting
- **Fallo al importar MetaTrader5**: `mt5_client.py` importa `MetaTrader5` al cargar el modulo, lo que requiere que el paquete este instalado. (ref: bot_trading/infrastructure/mt5_client.py)
- **Conexion MT5 fallida**: `MetaTrader5Client.connect()` lanza `MT5ConnectionError` si `mt5.initialize()` falla. (ref: bot_trading/infrastructure/mt5_client.py:MetaTrader5Client.connect)
- **Timeframe invalido**: `MetaTrader5Client.get_ohlcv()` lanza `ValueError` si el timeframe no esta en `TIMEFRAME_MAP`. (ref: bot_trading/infrastructure/mt5_client.py:MetaTrader5Client.get_ohlcv)
- **Datos CSV agotados**: `DevelopmentCsvDataProvider.get_data()` lanza `StopIteration` cuando no hay mas barras. (ref: bot_trading/infrastructure/data_fetcher.py:DevelopmentCsvDataProvider.get_data)
 - **Sin senales en estrategia**: `PivotZoneTestStrategy` retorna lista vacia si no puede calcular size (requiere equity y `broker_client._get_symbol_info`). `FakeBroker` ya expone estos campos en development; si se usa otro broker simulado hay que exponer `trade_tick_value/trade_tick_size` y volumenes minimos. (ref: bot_trading/application/strategies/pivot_zone_test_strategy.py:_calc_lot_size_by_stop, bot_trading/main.py:FakeBroker._get_symbol_info)
- **Ordenes bloqueadas por margen**: `RiskManager.check_margin_limits()` retorna False si el margen supera limites o no hay margen libre. (ref: bot_trading/application/risk_management.py:RiskManager.check_margin_limits)

## 10) Seguridad
- Archivos sensibles como `.env` y entornos virtuales se excluyen en `.gitignore`. (ref: .gitignore)
- `config.py` indica que credenciales MT5 deben venir de variables de entorno si `load_env_credentials=True`. (ref: config.py:_maybe_load_broker_env)
- El cliente MT5 opera sobre el terminal local instalado y no gestiona secretos externos por defecto. (ref: bot_trading/infrastructure/mt5_client.py:MetaTrader5Client.connect)

## 11) Tests
- Ejecutar toda la suite: `python -m pytest`. (ref: tests/*)
- Tests clave:
  - Motor y flujo basico: `tests/test_trading_bot.py`. (ref: tests/test_trading_bot.py)
  - Risk management: `tests/test_risk_management.py` y `tests/test_risk_management_integration.py`. (ref: tests/test_risk_management.py, tests/test_risk_management_integration.py)
  - Proveedores de datos: `tests/test_market_data_service.py`, `tests/test_production_data_provider.py`, `tests/test_development_csv_data_provider.py`. (ref: tests/test_market_data_service.py, tests/test_production_data_provider.py, tests/test_development_csv_data_provider.py)
  - Estrategia PivotZoneTest: `tests/test_pivot_zone_test_strategy.py`. (ref: tests/test_pivot_zone_test_strategy.py)
  - Cliente MT5 con mocks: `tests/test_meta_trader5_client_mocked.py`. (ref: tests/test_meta_trader5_client_mocked.py)
  - Componentes del visualizador M3: `tests/test_visualizer_components.py`. (ref: tests/test_visualizer_components.py)
- Nota: `tests/test_meta_trader5_client.py` contiene pruebas TDD de presencia de metodos y varios `pass`, no ejecuta MT5 real. (ref: tests/test_meta_trader5_client.py)

## 12) Roadmap
- Pendiente: no hay roadmap explicito en el repo actualmente. (ref: AGENTS.md, config.py, bot_trading/*, tests/*)

## 13) Changelog de logica
- Fecha: 2026-02-20
  - Cambios:
    - `PivotZoneTestStrategy` fija ATR interno en 14 para construir ancho de zona.
    - `n1` pasa a ser el multiplicador de distancia minima entre zonas guardadas (`min_distance = new_width * n1`).
    - `DevelopmentCsvDataProvider` ahora resuelve archivos CSV con variantes de nombre (`EURUSD.csv`, `EURUSD,.csv`, prefijos/sufijos).
  - Archivos:
    - bot_trading/application/strategies/pivot_zone_test_strategy.py
    - bot_trading/infrastructure/data_fetcher.py
    - tests/test_pivot_zone_test_strategy.py
    - README.md
  - Impacto esperado:
    - `n1` queda alineado con optimizacion de filtro de cercania entre zonas.
    - La paridad y replay development dejan de fallar por diferencias de naming en CSV.

- Fecha: 2026-02-18
  - Cambios:
    - `PivotZoneTestStrategy` endurece el filtro de cercania entre zonas: `min_distance` pasa de `new_width * 2.0` a `new_width * 3.0`.
  - Archivos:
    - bot_trading/application/strategies/pivot_zone_test_strategy.py
    - README.md
  - Impacto esperado:
    - Menor probabilidad de guardar zonas pivote demasiado juntas.

- Fecha: 2026-02-15
  - Cambios:
    - `FakeBroker.process_price_tick` mantiene cierre SL/TP inmediato en la vela actual, priorizando precio de `open` cuando hay gap.
    - Se añade `warmup` de 1 vela para SL/TP en posiciones recien abiertas (`_sltp_warmup_bars=1`) para alinear la activacion de brackets con Backtrader.
    - Cuando SL y TP se activan en la misma vela, `FakeBroker` emula doble fill y conserva una posicion invertida sintetica (paridad con Backtrader).
    - El motor (`TradingBot`) ya no fuerza `position=flat` en cierres sinteticos; usa `position_side/position_size` enviados por `FakeBroker`.
    - `PivotZoneTestStrategy` resetea estado interno al detectar reversion implicita de direccion en broker para no arrastrar brackets/trailing de la direccion previa.
    - Se conserva el comportamiento de `production` (MT5 real) sin cambios.
  - Archivos:
    - bot_trading/main.py
    - bot_trading/application/engine/bot_engine.py
    - bot_trading/application/strategies/pivot_zone_test_strategy.py
    - README.md
  - Impacto esperado:
    - `parity_report.md` sin mismatches dentro de tolerancias (`RUN_ID=20260215T185908Z`).

- Fecha: 2026-02-14
  - Cambios:
    - Se agrega visualizador incremental M3 con Bokeh para velas + zonas pivote + entradas/fills.
    - Nuevos modulos: `data_source`, `event_reader`, `state_store`, `plot_bokeh`, `runner` y entrypoint CLI `bot_trading/visualizer.py`.
    - `development` avanza ciclo a ciclo hasta terminar CSV; `production` queda en loop continuo.
    - Refresco del HTML configurable (default `900s`) y guardado robusto con `temp + replace` para tolerar archivo bloqueado en Windows sin detener el proceso.
    - Se agregan tests de componentes del visualizador (`tests/test_visualizer_components.py`).
  - Archivos:
    - bot_trading/visualization/data_source.py
    - bot_trading/visualization/event_reader.py
    - bot_trading/visualization/state_store.py
    - bot_trading/visualization/plot_bokeh.py
    - bot_trading/visualization/runner.py
    - bot_trading/visualizer.py
    - tests/test_visualizer_components.py
    - README.md
  - Impacto esperado:
    - Visualizacion operativa en M3 para seguimiento de ejecucion y zonas pivote sin bloquear ni frenar el bot por errores de escritura del HTML.

- Fecha: 2026-02-10
  - Cambios:
    - Se corrige el rechazo `10016 Invalid stops` en ordenes de mercado al normalizar `SL/TP` antes de `order_send`.
    - `MetaTrader5Client.send_market_order` ahora ajusta `SL/TP` usando `bid/ask`, `digits`, `trade_stops_level` y `trade_freeze_level`.
    - Si no existe un nivel positivo valido para `SL` o `TP`, ese campo se omite del request para evitar rechazo completo de la orden.
    - Se agregan tests de regresion para validar ajuste de `SL/TP` en BUY y omision de `SL` imposible.
  - Archivos:
    - bot_trading/infrastructure/mt5_client.py
    - tests/test_meta_trader5_client_mocked.py
    - README.md
  - Impacto esperado:
    - Menos rechazos en MT5 por `Invalid stops` durante ejecucion real y mayor robustez frente a spreads/distancias minimas del simbolo.

- Fecha: 2026-02-06
  - Cambios:
    - Se corrige el error repetitivo `can't compare offset-naive and offset-aware datetimes` en produccion durante la descarga incremental de OHLCV.
    - `MetaTrader5Client.get_ohlcv()` ahora normaliza `start/end` a UTC-aware y parsea el indice OHLCV con `utc=True`.
    - `ProductionDataProvider` normaliza `now`, cache de base e `incremental_start` a UTC en todo el flujo incremental.
    - Se agrega test de regresion para rango mixto naive/aware en `tests/test_meta_trader5_client_mocked.py`.
  - Archivos:
    - bot_trading/infrastructure/mt5_client.py
    - bot_trading/infrastructure/data_fetcher.py
    - tests/test_meta_trader5_client_mocked.py
    - README.md
  - Impacto esperado:
    - El bot deja de repetir errores de obtencion de datos por timezone y mantiene un manejo consistente en UTC para produccion y desarrollo.

- Fecha: 2026-02-06
  - Cambios:
    - Se elimina `lot_size` de `SymbolConfigEntry` y `SymbolConfig` porque el sizing real de `PivotZoneTestStrategy` se calcula por riesgo y distancia a stop.
    - Se eliminan campos de configuracion sin uso en runtime: `BrokerConfig.slippage_points`, `BrokerConfig.filling_mode`, `DataConfig.resample_timeframes` y `DataConfig.history_minutes`.
    - Se simplifica el mapeo de simbolos en `bot_trading/main.py` y se actualizan tests/documentacion al nuevo contrato de configuracion.
  - Archivos:
    - config.py
    - bot_trading/domain/entities.py
    - bot_trading/main.py
    - tests/test_entities.py

- Fecha: 2026-02-06
  - Cambios:
    - Se ajusta el sizing de `PivotZoneTestStrategy` para evitar lotajes excesivos en live: ademas del riesgo por stop, ahora se aplica un tope de exposicion nocional en torno a `size_pct` del equity.
    - Se mantiene el calculo por distancia a stop como base, pero el lote final se limita por cap nocional cuando `trade_contract_size` esta disponible en `symbol_info`.
    - Si el cap nocional queda por debajo del volumen minimo del simbolo, la estrategia descarta la entrada (no fuerza un lote por encima del cap).
    - Se agrega test de regresion para validar el cap nocional y evitar tamanos como `46.46` lotes con `size_pct=0.05`.
  - Archivos:
    - bot_trading/application/strategies/pivot_zone_test_strategy.py
    - tests/test_pivot_zone_test_strategy.py
    - README.md
  - Impacto esperado:
    - Menos rechazos por margen insuficiente y tamanos de posicion mas coherentes con la intencion de exponer aproximadamente `size_pct` del capital por trade.

- Fecha: 2026-02-07
  - Cambios:
    - Se corrige el manejo de ordenes pendientes TP/SL en MT5 para reducir rechazos `10015 Invalid price` en `SELL_STOP/BUY_STOP/SELL_LIMIT/BUY_LIMIT`.
    - `MetaTrader5Client.create_pending_order` ahora normaliza el precio solicitado contra `bid/ask` y la distancia minima del simbolo (`trade_stops_level` / `trade_freeze_level`).
    - `MetaTrader5Client.modify_order` ahora:
      - normaliza precio cuando aplica,
      - omite la modificacion si el precio ya coincide (evita ruido),
      - trata `10025 No changes` como resultado exitoso en lugar de error operativo.
    - Se agregan tests mockeados de regresion para ambos casos.
  - Archivos:
    - bot_trading/infrastructure/mt5_client.py
    - tests/test_meta_trader5_client_mocked.py
    - README.md
  - Impacto esperado:
    - Menos fallos al crear/actualizar SL dinamico como orden pendiente y menor ruido de errores repetitivos en logs.

- Fecha: 2026-02-07
  - Cambios:
    - Se corrige la sobre-apertura de posiciones en cuentas MT5 tipo `Hedge` cuando la estrategia usaba pendientes opuestas como bracket.
    - `PivotZoneTestStrategy` desactiva la creacion de pendientes opuestas (TP/SL) en `MetaTrader5Client` para evitar que un `SELL_STOP/BUY_STOP` abra una posicion nueva en vez de cerrar.
    - En modo MT5 real, ahora se limpian pendientes remanentes del mismo `symbol+magic` en cada ciclo aunque no haya posicion, para eliminar `placed` huerfanas de ejecuciones anteriores.
    - En modo MT5 real, la estrategia limpia pendientes heredadas de bracket y actualiza trailing stop via SL/TP de posicion (`modify_position_sl_tp`) cuando el broker lo soporta.
    - Se agrega `MetaTrader5Client.modify_position_sl_tp` (MT5 `TRADE_ACTION_SLTP`).
    - Se agrega test de regresion para asegurar que en modo MT5 real no se crean brackets opuestos y se limpian pendientes.
  - Archivos:
    - bot_trading/application/strategies/pivot_zone_test_strategy.py
    - bot_trading/infrastructure/mt5_client.py
    - tests/test_pivot_zone_test_strategy.py
    - README.md
  - Impacto esperado:
    - La estrategia mantiene una sola posicion por simbolo para su magic number en ejecución live y evita acumulación de posiciones opuestas por brackets.
    - README.md
  - Impacto esperado:
    - Menor deuda de configuracion y menos parametros huérfanos; la configuracion refleja mejor el comportamiento real del bot.

- Fecha: 2026-02-05
  - Cambios:
    - `FakeBroker` admite cerrar SL/TP solo al close (`PARITY_CLOSE_ON_CLOSE=1`) y mantiene la opcion de diferir cierre con `PARITY_CLOSE_DELAY=1` para paridad.
    - El motor fuerza un flush de cierres pendientes al agotar CSV para no perder eventos de cierre.
    - `PivotZoneTestStrategy` permite omitir la validacion de SL/TP en paridad con `PARITY_SKIP_SLTP_VALIDATION=1`.
    - El motor puede registrar cierres SL/TP usando el precio de cierre (`PARITY_CLOSE_PRICE_ON_CLOSE=1`) para alinear eventos de paridad.
    - El motor actualiza el estado interno de posiciones al procesar cierres SL/TP para permitir re-entradas en la misma vela.
    - OrderExecutor puede limpiar posiciones sin emitir eventos extra de `position_close` cuando el cierre ya se registro por paridad.
  - Archivos:
    - bot_trading/main.py
    - bot_trading/application/engine/bot_engine.py
    - bot_trading/application/strategies/pivot_zone_test_strategy.py
    - README.md
  - Impacto esperado:
    - Menos desalineacion en bar_index/ordenes y menos cascadas de mismatches en parity_report.

- Fecha: 2026-01-31
  - Cambios:
    - El resample local usa etiqueta en cierre (label/closed right) para alinear el bot con el backtest.
    - Pivot3Candle confirma pivotes con las 3 velas previas (retardo de 1 vela).
  - Archivos:
    - bot_trading/infrastructure/data_fetcher.py
    - bot_trading/application/strategies/pivot_zone_test_strategy.py
    - README.md
  - Impacto esperado:
    - Los timeframes resampleados quedan alineados al cierre del bloque y los pivotes se confirman con el mismo retraso que el backtest.

- Fecha: 2026-01-31
  - Cambios:
    - FakeBroker expone eventos de cierre por SL/TP y el motor emite order_fill/position en development para paridad con backtest.
    - El trailing stop actualiza stop_loss en posiciones del FakeBroker para que los cierres usen el nivel actualizado.
  - Archivos:
    - bot_trading/main.py
    - bot_trading/application/engine/bot_engine.py
    - bot_trading/application/strategies/pivot_zone_test_strategy.py
    - README.md
  - Impacto esperado:
    - Paridad mas cercana en cierres SL/TP entre bot y backtest.

- Fecha: 2026-01-30
  - Cambios:
    - En `development` (CSV), los timeframes resampleados (p.ej. `M3`) descartan la Ãºltima vela parcial para evitar OHLC inestables que cambian con cada nueva vela `M1`.
    - `PivotZoneTestStrategy` ajusta la confirmaciÃ³n de pivotes 3 velas para alinear el timing con el backtest (sin adelantar pivotes usando la vela actual).
    - Se aÃ±ade instrumentaciÃ³n JSONL `event_type=pivot_confirmed` en TF_zone para depurar paridad.
  - Archivos:
    - bot_trading/infrastructure/data_fetcher.py
    - bot_trading/application/strategies/pivot_zone_test_strategy.py
    - README.md
  - Impacto esperado:
    - Menos divergencias entre development (CSV) y producciÃ³n/backtest al evitar consumir velas superiores "en formaciÃ³n"; los pivotes reportados deberÃ­an coincidir.
- Fecha: 2026-01-29
  - Cambios:
    - `LoggingConfig` expone `pivot_log_file_path` para configurar la ruta del log dedicado de zonas pivote.
    - `_configure_logging` usa la ruta configurable en lugar de un path hardcodeado.
    - README actualizado con los paths de logging por entorno y el nuevo campo.
  - Archivos:
    - config.py
    - bot_trading/main.py
    - README.md
  - Impacto esperado:
    - Se puede redirigir el log de zonas pivote por entorno desde `config.py` sin tocar el código del bot.
- Fecha: 2026-01-29
  - Cambios:
    - `pivot_zones.log` registra top/bot/mid/width con 8 decimales para evitar redondeo prematuro en pares forex.
    - El log de consola `ZONA_GUARDADA` usa la misma precisión que el archivo dedicado para mantener coherencia.
  - Archivos:
    - bot_trading/application/strategies/pivot_zone_test_strategy.py
    - README.md
  - Impacto esperado:
    - Mayor trazabilidad de zonas pivote sin perder pipettes al inspeccionar los logs.
- Fecha: 2026-01-29
  - Cambios:
    - Se habilita configuracion por simbolo (n1/n2/n3/size_pct/p) reutilizada en los tres entornos mediante `DEFAULT_SYMBOLS`/`DEFAULT_STRATEGIES`.
    - Las estrategias completan `allowed_symbols` con todos los simbolos configurados si se deja vacio.
    - PivotZoneTest aplica overrides por simbolo y los tests cubren sizing y overrides.
  - Archivos:
    - config.py
    - bot_trading/main.py
    - bot_trading/domain/entities.py
    - bot_trading/application/strategies/pivot_zone_test_strategy.py
    - tests/test_pivot_zone_test_strategy.py
    - README.md
  - Impacto esperado:
    - Se pueden definir parametros distintos por activo sin duplicar estrategias ni entornos; los tres modos comparten flujo y solo difieren en broker/datos/logging/riesgo.
- Fecha: 2026-01-29
  - Cambios:
    - Se añade un logger dedicado (`pivot_zones`) que persiste cada zona pivote consolidada en `logs/pivot_zones.log` con datos de símbolo/timeframe/top/bot/width.
    - El log usa el timestamp de la vela (datos backtest/stream) en lugar de la hora de ejecución del bot.
    - Los archivos logs/development.log y logs/pivot_zones.log se truncan en cada arranque para contener sólo la ejecución actual.
  - Archivos:
    - bot_trading/main.py
    - bot_trading/application/strategies/pivot_zone_test_strategy.py
    - README.md
  - Impacto esperado:
    - Mayor trazabilidad de la generación de zonas pivote por activo sin mezclar con otros eventos del bot.
- Fecha: 2026-01-26
  - Cambios:
    - `PivotZoneTestStrategy` deja de descartar sistemáticamente la última vela de `tf_entry`; ahora consume todas las velas cerradas para alinearse con el backtest de Backtrader.
    - Se ajustan los datos de prueba de ruptura para respetar el filtro de origen incluyendo la vela más reciente.
  - Archivos:
    - bot_trading/application/strategies/pivot_zone_test_strategy.py
    - tests/test_pivot_zone_test_strategy.py
    - README.md
  - Impacto esperado:
    - Timing de ruptura y filtros idénticos entre backtest y bot en producción/desarrollo; los logs deberían reflejar las mismas entradas.
- Fecha: 2026-01-23
  - Cambios:
    - `PivotZoneTestStrategy` registra logs al generar señal de entrada (price/size/SL/TP) y al actualizar trailing stop (nuevo SL).
  - Archivos:
    - bot_trading/application/strategies/pivot_zone_test_strategy.py
    - README.md
  - Impacto esperado:
    - Mayor trazabilidad en consola de entradas y ajustes de stop.
- Fecha: 2026-01-23
  - Cambios:
    - Se añade log explícito al crear una zona pivote (`ZONA_GUARDADA`) con bar/top/bot para seguimiento en terminal.
  - Archivos:
    - bot_trading/application/strategies/pivot_zone_test_strategy.py
    - README.md
  - Impacto esperado:
    - Mayor trazabilidad en consola cuando se crean zonas.
- Fecha: 2026-01-23
  - Cambios:
    - `PivotZoneTestStrategy` deja de forzar `tf_stop=tf_zone`; los timeframes se respetan tal como se definen en `config.py`.
    - Se elimina `use_same_zone_stop_tf` de la configuración y de la estrategia.
  - Archivos:
    - bot_trading/application/strategies/pivot_zone_test_strategy.py
    - config.py
    - bot_trading/main.py
    - tests/test_pivot_zone_test_strategy.py
    - README.md
  - Impacto esperado:
    - `tf_stop` puede ser independiente de `tf_zone` en todos los entornos.
- Fecha: 2026-01-23
  - Cambios:
    - `PivotZoneTestStrategy` alinea la construcción/lock de zonas con el indicador PivotZone del backtest (sin alternancia legacy) y guarda la zona solo en transición valid 0→1.
    - Se corrige el campo `bar` de la zona guardada para usar índice de barra (no precio).
    - Tests de la estrategia ajustados a la lógica del indicador.
  - Archivos:
    - bot_trading/application/strategies/pivot_zone_test_strategy.py
    - tests/test_pivot_zone_test_strategy.py
    - README.md
  - Impacto esperado:
    - Mayor consistencia entre backtest Backtrader y el port en el bot (zonas y timing de guardado).
- Fecha: 2026-01-23
  - Cambios:
    - `FakeBroker` guarda el ultimo close y simula fills de SL/TP via `process_price_tick`, eliminando posiciones y registrando trades cerrados en modo paper.
    - `TradingBot.run_once` llama a `process_price_tick` por simbolo usando el close mas reciente del timeframe base para que el loop de desarrollo refleje cierres y permita nuevas entradas.
    - `PivotZoneTestStrategy` alinea el lock de zonas con el Backtrader original: requiere que el precio haya estado por encima y por debajo de la zona provisional (o alternancia legacy), reduciendo divergencias entre backtest y modo desarrollo.
    - En `development`, `tf_zone` vuelve a `M3` y `tf_stop` queda en `M1` para igualar la construcción de zonas del backtest pero seguir usando pivotes rápidos para el stop.
    - Documentacion actualizada para reflejar el nuevo comportamiento en development.
  - Archivos:
    - bot_trading/main.py
    - bot_trading/application/engine/bot_engine.py
    - bot_trading/application/strategies/pivot_zone_test_strategy.py
    - README.md
    - config.py
  - Impacto esperado:
    - En modo development con CSV se veran multiples trades (como en el backtest) porque las posiciones ahora se cierran al tocar SL/TP simulados.
- Fecha: 2026-01-21
  - Cambios:
    - `skip_sleep_when_simulated` queda en True por defecto para el entorno `development`, asi las simulaciones con CSV avanzan sin dormir.
    - Se agrega una prueba que asegura el default, otra que verifica que el flag en False obliga a dormir, y se documenta el nuevo comportamiento en README.
    - `config.load_settings()` devuelve una copia para evitar que mutaciones previas de `settings` ignoren cambios en el toggle.
  - Archivos:
    - config.py
    - tests/test_development_csv_data_provider.py
    - README.md
  - Impacto esperado:
    - El loop en development corre en modo acelerado por defecto; si se desea la cadencia real se puede poner el flag en False en `config.py`.
- Fecha: 2026-01-21
  - Cambios:
    - FakeBroker alinea balance/equity/margin_free a 20,000 para reflejar el capital de riesgo de referencia.
    - skip_sleep_when_simulated ahora falla rapido si se habilita sin reloj simulado, evitando caer silenciosamente al modo con sleeps.
  - Archivos:
    - bot_trading/main.py
    - bot_trading/application/engine/bot_engine.py
    - README.md
  - Impacto esperado:
    - El modo development usa el mismo capital configurado y el toggle de simulacion acelerada se respeta o alerta cuando falta reloj simulado.
- Fecha: 2026-01-20
  - Cambios:
    - Produccion y desarrollo usan PivotZoneTest con M1/M3 tanto para zona como para stop, alineando timeframes y resample; development solo cambia el feed CSV.
    - `skip_sleep_when_simulated` pasa a ser False por defecto para compartir la misma cadencia real; se puede activar manualmente para pruebas rapidas con datos ficticios.
    - `TradingBot.run_synchronized` actualiza el default y el test de proveedor CSV invoca explicitamente el modo acelerado.
    - README documenta el flujo primario con la nueva configuracion unificada.
  - Archivos:
    - config.py
    - bot_trading/application/engine/bot_engine.py
    - tests/test_development_csv_data_provider.py
    - README.md
  - Impacto esperado:
    - Development replica el flujo de produccion (riesgo, estrategias, loop) cambiando solo la fuente de datos, con opcion de simulacion acelerada bajo demanda.
- Fecha: 2026-01-05
  - Cambios:
    - `config.py` añade el toggle `skip_sleep_when_simulated` en `LoopConfig` para controlar si se evita el sleep con CSV.
    - `TradingBot.run_synchronized` acepta el flag y respeta la configuración para decidir si usa timestamps simulados sin esperar.
    - `README.md` documenta el nuevo comportamiento y la bandera de configuración.
    - `bot_trading/main.py` propaga la configuración al motor.
  - Archivos:
    - config.py
    - bot_trading/application/engine/bot_engine.py
    - bot_trading/main.py
    - README.md
  - Impacto esperado:
    - Permite elegir desde `config.py` si el bucle en modo desarrollo (CSV) avanza sin `sleep` o se sincroniza con reloj real, evitando sorpresas al probar.
- Fecha: 2026-01-04
  - Cambios:
    - `config.py` pasa a ser fuente unica de configuracion y valida antes de ejecutar.
    - `bot_trading/main.py` consume `settings` y permite lanzar el bot desde `config.py`.
    - Logging y seleccion de broker/datos/riesgo se controlan via config declarativa.
  - Archivos:
    - config.py
    - bot_trading/main.py
    - README.md
  - Impacto esperado:
    - Ejecucion centralizada desde `config.py` sin editar `main.py`; ajustes de entorno/broker/logging se hacen en un solo lugar.
- Fecha: 2026-01-04
  - Cambios:
    - `FakeBroker` ahora simula `_get_symbol_info` y ordenes pendientes para habilitar calculo de lotes y brackets en modo development.
    - La estrategia PivotZoneTest puede emitir senales en development usando CSV al contar con sizing valido.
  - Archivos:
    - bot_trading/main.py
    - README.md
  - Impacto esperado:
    - El modo development genera senales y brackets con datos CSV sin broker real; menos falsos negativos por falta de info del simbolo.

- Fecha: 2026-02-24
  - Cambios:
    - `PivotZoneTestStrategy` simplifica entrada: mantiene origen `2/3` dentro + `2 cierres` fuera y elimina filtro `close near extreme` y `follow-through`.
    - El `stop inicial` ahora se toma como el ultimo pivote confirmado de `TF_stop` que cae dentro de la zona rota (LONG=min, SHORT=max).
    - Se desactiva el trailing estructural en gestion de posicion: salida solo por `SL inicial` o `TP` de zona objetivo.
    - Se ajustan tests de estrategia para validar el nuevo comportamiento (sin trailing y stop dentro de zona).
  - Archivos:
    - bot_trading/application/strategies/pivot_zone_test_strategy.py
    - tests/test_pivot_zone_test_strategy.py
    - README.md
  - Impacto esperado:
    - Menos filtros de entrada, manteniendo ruptura confirmada.
    - SL inicial anclado a pivote interno de zona rota.
    - Gestion mas simple y estable (sin desplazamiento de stop).
