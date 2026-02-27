# ====================================================================================
# Flujo principal
# ====================================================================================

import os
import pandas as pd
import config
from typing import List, Optional
from data import DataManager, ensure_dirs, erase_directory, safe_name, output_path
from backtest import Backtest, OptimizeCore, Grid, Heatmaps, Stats, Trades, Plots, Results, CSCV as CSCVSave, WF as WFSave
from strategies import StrategyMultiAsset, make_multi 

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.dirname(os.path.abspath(config.__file__))


def _resolve_data_folder(folder_name: str) -> Optional[str]:
    """
    Devuelve la ruta absoluta válida para una carpeta de datos.
    Prueba la ruta tal cual, luego relativa al proyecto y a la carpeta de config.
    """
    if not folder_name:
        return None

    candidates = []
    trimmed = folder_name.strip()
    if os.path.isabs(trimmed):
        candidates.append(trimmed)
    else:
        candidates.append(os.path.abspath(trimmed))  # relativo al CWD actual
        candidates.append(os.path.join(PROJECT_ROOT, trimmed))
        candidates.append(os.path.join(CONFIG_DIR, trimmed))

    for candidate in candidates:
        candidate_norm = os.path.normpath(candidate)
        if os.path.isdir(candidate_norm):
            return candidate_norm

    return None


def discover_data_folders() -> List[str]:
    """
    Busca los atributos DATA_FOLDER_XX en config y devuelve las carpetas existentes.
    """
    data_folders: List[str] = []
    for i in range(1, 100):
        folder_attr = f"DATA_FOLDER_{i:02d}"
        if not hasattr(config, folder_attr):
            continue
        folder_value = getattr(config, folder_attr)
        resolved = _resolve_data_folder(folder_value) if folder_value else None
        if resolved:
            data_folders.append(resolved)
            print(f"[MAIN] Carpeta de datos detectada: {resolved}")
        elif folder_value:
            print(
                f"[MAIN][WARN] No se encontró '{folder_value}' ni en {os.getcwd()} "
                f"ni en {PROJECT_ROOT}"
            )
    return data_folders

def load_strategies_from_results(results_file: str = "results/summary_stats.xlsx"):
    """
    Lee results.xlsx y crea estrategias StrategyMultiAsset con parámetros encontrados.
    
    Devuelve tupla: (strategy_cls, params_by_asset_dict)
    donde params_by_asset_dict mapea asset_name -> params_dict
    """
    import os
    
    if not os.path.isabs(results_file):
        results_file = output_path(results_file)

    if not os.path.exists(results_file):
        print(f"[MAIN][WARN] Archivo {results_file} no existe")
        return []
    
    try:
        df = pd.read_excel(results_file)
        print(f"[MAIN] Leyendo {len(df)} filas desde {results_file}")
        
        strategies_config = []
        
        # Obtener estrategias base desde config
        base_strategies = getattr(config, "STRATEGIES", [])
        
        # Crear diccionario para mapear nombres a clases base (inner_cls)
        strategy_map = {}
        for strat_cls in base_strategies:
            if issubclass(strat_cls, StrategyMultiAsset):
                inner_cls = getattr(strat_cls, 'inner_cls', None)
                if inner_cls:
                    strategy_map[strat_cls.__name__] = inner_cls
                    strategy_map[inner_cls.__name__] = inner_cls
        
        if not strategy_map:
            print("[MAIN][WARN] No se encontraron estrategias StrategyMultiAsset en config.STRATEGIES")
            return []
        
        # Agrupar por tipo de estrategia
        strategies_by_type = {}
        for idx, row in df.iterrows():
            source = str(row.get('__source__', ''))
            inner_cls = None
            
            for name, cls in strategy_map.items():
                if name in source:
                    inner_cls = cls
                    break
            
            if not inner_cls:
                continue
            
            # Extraer parámetros n1, n2, n3 y Avg. Drawdown [%]
            params = {}
            param_names = ['n1', 'n2', 'n3']
            for param_name in param_names:
                if param_name in row.index:
                    value = row[param_name]
                    if pd.notna(value):
                        try:
                            params[param_name] = int(value)
                        except (ValueError, TypeError):
                            params[param_name] = value
            
            # Extraer Avg. Drawdown [%] si está disponible
            if 'Avg. Drawdown [%]' in row.index:
                avg_dd = row['Avg. Drawdown [%]']
                if pd.notna(avg_dd):
                    try:
                        params['avg_drawdown'] = float(avg_dd)
                    except (ValueError, TypeError):
                        params['avg_drawdown'] = avg_dd
            
            asset_name = row.get('Asset', None) if 'Asset' in row.index else None
            
            if asset_name:
                strat_key = inner_cls.__name__
                if strat_key not in strategies_by_type:
                    strategies_by_type[strat_key] = {'inner_cls': inner_cls, 'params_by_asset': {}}
                strategies_by_type[strat_key]['params_by_asset'][asset_name] = params
                print(f"[MAIN] Parámetros para {strat_key} - {asset_name}: {params}")
        
        # Crear estrategias configuradas
        for strat_key, config_data in strategies_by_type.items():
            inner_cls = config_data['inner_cls']
            params_by_asset = config_data['params_by_asset']
            
            # Crear estrategia wrapper (usará parámetros por defecto, luego se sobrescribirán)
            configured_strategy = make_multi(inner_cls)
            
            strategies_config.append((configured_strategy, params_by_asset))
        
        return strategies_config
        
    except Exception as e:
        print(f"[MAIN][ERROR] Error leyendo {results_file}: {e}")
        import traceback
        traceback.print_exc()
        return []

def main() -> None:
    """
    Flujo principal del programa:

      1) Preparar carpetas de salida (plots, stats, trades, etc.).
      2) Cargar CSVs de cada carpeta habilitada (DATA_FOLDER_01, DATA_FOLDER_02, ...)
      3) Por cada posición i (ordenada alfabéticamente):
         - Construir df_master = [data01[i], data02[i], data03[i], ...]
         - Por cada estrategia en config.STRATEGIES:
           a) Optimizar optimize()
           b) Guardar grid.xlsx con nombre Estrategia_Activo.xlsx
           c) Guardar heatmaps.html
           d) Seleccionar mejores parámetros
           e) Guardar stats, trades, plots con nombre Estrategia_Activo.xlsx
      4) Consolidar results.xlsx con todos los stats

    Logs ultra detallados para seguimiento paso a paso.
    """
    # ------------------------------------------------------------------
    # 1) Preparar carpeta de resultados
    # ------------------------------------------------------------------
    ensure_dirs()
    erase_directory()
    print("\n==============================================================")
    print("[MAIN] Iniciando flujo principal de backtests")
    print("==============================================================")

    # ------------------------------------------------------------------
    # 2) Descubrir carpetas de datos habilitadas en config.py
    # ------------------------------------------------------------------
    data_folders = discover_data_folders()

    if not data_folders:
        print("[MAIN][ERROR] No se encontraron carpetas de datos habilitadas en config.py")
        return
    
    print(f"[MAIN] Total de carpetas de datos: {len(data_folders)}")

    # ------------------------------------------------------------------
    # 3) Cargar CSVs de cada carpeta en listas separadas
    # ------------------------------------------------------------------
    dm = DataManager(data_root=PROJECT_ROOT, tz=getattr(config, "TIMEZONE", "UTC"))
    
    print("\n[MAIN] Cargando CSVs desde carpetas...")
    
    # Lista de listas: [[df01_asset1, df01_asset2, ...], [df02_asset1, df02_asset2, ...], ...]
    folders_dfs = []
    folders_names = []
    
    for folder in data_folders:
        print(f"\n[MAIN] Procesando carpeta: {folder}")
        dfs_in_folder = dm.load_all_from_folder(folder)
        
        if not dfs_in_folder:
            print(f"[MAIN][WARN] Carpeta {folder} no contiene CSVs válidos")
            folders_dfs.append([])
            folders_names.append([])
            continue
        
        # Extraer nombres de archivos (sin extensión)
        names = []
        for df in dfs_in_folder:
            # El nombre viene del atributo ._asset_name asignado en DataManager.load_tradingview
            name = getattr(df, '_asset_name', f'asset_{len(names)}')
            names.append(name)
        
        folders_dfs.append(dfs_in_folder)
        folders_names.append(names)
        print(f"[MAIN]   {len(dfs_in_folder)} archivo(s) cargados: {names}")
    
    # Verificar que todas las carpetas tengan el mismo número de archivos
    num_assets = len(folders_dfs[0]) if folders_dfs else 0
    
    if num_assets == 0:
        print("[MAIN][ERROR] No se cargaron datos válidos")
        return
    
    for idx, dfs_list in enumerate(folders_dfs[1:], start=1):
        if len(dfs_list) != num_assets:
            print(f"[MAIN][WARN] Carpeta {data_folders[idx]} tiene {len(dfs_list)} archivos, "
                  f"esperaba {num_assets}. Rellenando con None.")
            # Rellenar con None si falta algún archivo
            while len(dfs_list) < num_assets:
                dfs_list.append(None)
    
    print(f"\n[MAIN] Total de activos a evaluar: {num_assets}")

    # ------------------------------------------------------------------
    # 4) Configurar motor de backtest y optimización
    # ------------------------------------------------------------------
    broker_cfg = {
        "cash": float(getattr(config, "CASH", 100000.0)),
        "commission": float(getattr(config, "COMMISSION", 0.0)),
        "margin": 1.0 / max(1.0, float(getattr(config, "MARGIN", 1))),
        "trade_on_close": bool(getattr(config, "TRADE_ON_CLOSE", True))
    }
    
    backtest = Backtest(broker_cfg=broker_cfg)
    optimizer = OptimizeCore(maximize_metric=getattr(config, "OPTIMIZE_MAXIMIZE", "Sharpe Ratio"))
    
    print("\n[MAIN] Configuración del broker:")
    print(f"  - Cash inicial: ${broker_cfg['cash']:,.2f}")
    print(f"  - Comisión: {broker_cfg['commission']*100:.3f}%")
    print(f"  - Margen (apalancamiento): {1.0/broker_cfg['margin']:.1f}x")
    print(f"  - Trade on close: {broker_cfg['trade_on_close']}")

    # ------------------------------------------------------------------
    # 5) Obtener estrategias
    # ------------------------------------------------------------------
    strategies = getattr(config, "STRATEGIES", [])
    
    if not strategies:
        print("[MAIN][WARN] No hay estrategias definidas en config.STRATEGIES")
        return
    
    print(f"\n[MAIN] Estrategias a ejecutar: {len(strategies)}")
    for strat_cls in strategies:
        print(f"  - {strat_cls.__name__}")
    # Detectar modo multi: multi-asset O multi-timeframe
    is_multi_run = all(
        issubclass(strat_cls, StrategyMultiAsset) or getattr(strat_cls, '_is_multi_tf', False)
        for strat_cls in strategies
    )
    print(f"[MAIN] Modo multi-activo: {is_multi_run}")

    # ------------------------------------------------------------------
    # 6) BUCLE PRINCIPAL: Por cada activo (posición i)
    # ------------------------------------------------------------------
    for asset_idx in range(num_assets):
        asset_name = folders_names[0][asset_idx] if folders_names[0] else f"asset_{asset_idx}"
        
        print("\n" + "="*70)
        print(f"[MAIN] ACTIVO [{asset_idx+1}/{num_assets}]: {asset_name}")
        print("="*70)
        
        # Construir df_master para este activo (tomar posición i de cada carpeta)
        df_master = []
        for folder_idx, dfs_list in enumerate(folders_dfs):
            if asset_idx < len(dfs_list) and dfs_list[asset_idx] is not None:
                df_master.append(dfs_list[asset_idx])
            else:
                print(f"[MAIN][WARN] No hay datos en carpeta {data_folders[folder_idx]} "
                      f"para posición {asset_idx}")
        
        if not df_master:
            print(f"[MAIN][WARN] No hay datos para {asset_name}, saltando...")
            continue
        
        print(f"\n[MAIN] df_master construido con {len(df_master)} DataFrame(s):")
        for idx, df in enumerate(df_master):
            print(f"  df_master[{idx}]: {len(df)} filas, {df.index[0]} -> {df.index[-1]}")
        
        # Sincronizar si hay múltiples DataFrames (sólo en modo single-asset)
        if len(df_master) > 1:
            if is_multi_run:
                print("[MAIN] Modo multi-activo activo: se usan datos originales sin intersección")
            else:
                print(f"\n[MAIN] Sincronizando {len(df_master)} DataFrames...")
                df_master, _ = dm.sync_multi_asset(df_master, drop_mismatched=True)
                print(f"[MAIN] Sincronización OK. Filas comunes: {len(df_master[0])}")
        
        # ----------------------------------------------------------
        # 7) Por cada estrategia
        # ----------------------------------------------------------
        for strat_idx, strat_cls in enumerate(strategies):
            print("\n" + "-"*70)
            print(f"[MAIN] Estrategia [{strat_idx+1}/{len(strategies)}]: "
                  f"{strat_cls.__name__} sobre {asset_name}")
            print("-"*70)
            
            # Nombre de salida base: Estrategia_Activo
            base_name = f"{strat_cls.__name__}_{safe_name(asset_name)}"
            
            # ----------------------------------------------------------
            # 7a) Optimización
            # ----------------------------------------------------------
            print(f"\n[MAIN] Ejecutando optimización...")
            
            param_cols = list(getattr(config, "param_cols", ['n1', 'n2', 'n3']))
            
            grid_df = optimizer.optimize(
                backtest=backtest,
                strategy_cls=strat_cls,
                df_master=df_master,
                param_cols=param_cols,
                ranges=None
            )
            
            if grid_df.empty:
                print(f"[MAIN][WARN] Grid vacío para {base_name}, saltando...")
                continue
            
            print(f"[MAIN] Optimización completada. Grid: {len(grid_df)} combinaciones")
            
            # ----------------------------------------------------------
            # 7b) Guardar grid.xlsx
            # ----------------------------------------------------------
            print(f"\n[MAIN] Guardando grid.xlsx...")
            Grid.save(grid_df, strategy_name=f"{base_name}_grid")
            
            # ----------------------------------------------------------
            # 7h) (Opcional) Análisis de robustez: CSCV
            # ----------------------------------------------------------
            cscv_pbo_value = None
            if getattr(config, "USE_CSCV", False):
                print(f"\n[MAIN] Ejecutando análisis CSCV para {strat_cls.__name__}...")
                
                try:
                    from overfitting import CSCV
                    
                    # Crear instancia de CSCV con backtest
                    cscv_runner = CSCV(backtest=backtest)
                    
                    # Validar que df_master tenga datos
                    if not df_master or len(df_master) == 0:
                        print(f"[MAIN][WARN] No hay datos en df_master para CSCV, saltando...")
                    elif df_master[0] is None or df_master[0].empty:
                        print(f"[MAIN][WARN] El primer DataFrame de df_master está vacío para CSCV, saltando...")
                    else:
                        # Configuración opcional para CSCV (puedes ajustar según necesites)
                        cscv_cfg = {
                            'n_partitions': 10,
                            'top_k': 10,
                            'metric': getattr(config, "OPTIMIZE_MAXIMIZE", "Sharpe Ratio"),
                            'selection_mode': 'diversified',
                            'top_percent_pool': 0.30
                        }
                        
                        # Ejecutar CSCV - pasar toda la lista df_master (soporta 1 activo y multi-activo)
                        cscv_df = cscv_runner.run(
                            metrics_df=grid_df,
                            df_master=df_master,  # Lista completa de DataFrames
                            strategy_cls=strat_cls,
                            param_cols=param_cols,
                            cfg=cscv_cfg
                        )
                        
                        # Generar resumen
                        cscv_summary = CSCV.summarize(cscv_df)

                        # Guardar resultados CSCV
                        CSCVSave.save(cscv_df, strategy_name=f"{base_name}_cscv")

                        # Extraer valor PBO
                        if not cscv_summary.empty and 'CSCV_PBO' in cscv_summary.columns:
                            cscv_pbo_value = float(cscv_summary['CSCV_PBO'].values[0])
                            print(f"[MAIN] CSCV completado. PBO: {cscv_pbo_value:.3f}")
                            print(f"[MAIN] Resultados CSCV guardados para {base_name}")
                        else:
                            print(f"[MAIN][WARN] No se pudo extraer CSCV_PBO del resumen")
                        
                except ImportError:
                    print("[MAIN][WARN] Módulo 'overfitting.py' no encontrado, saltando CSCV")
                except Exception as e:
                    print(f"[MAIN][ERROR] Error en CSCV para {strat_cls.__name__}: {e}")

            # ------------------------------------------------------------------
            # 9) (Opcional) Análisis de robustez: Stress Test
            # ------------------------------------------------------------------
            if getattr(config, "USE_STRESS", False):
                print(f"\n[MAIN] Ejecutando Stress Test para {strat_cls.__name__}...")
                
                try:
                    from overfitting import Stress
                    
                    # Crear instancia de Stress con backtest
                    stress_runner = Stress(backtest=backtest)
                    
                    # Validar que df_master tenga datos
                    if not df_master or len(df_master) == 0:
                        print(f"[MAIN][WARN] No hay datos en df_master para Stress Test, saltando...")
                    elif df_master[0] is None or df_master[0].empty:
                        print(f"[MAIN][WARN] El primer DataFrame de df_master está vacío para Stress Test, saltando...")
                    elif grid_df.empty:
                        print(f"[MAIN][WARN] Grid vacío para Stress Test, saltando...")
                    else:
                        # Configuración para Stress Test
                        stress_cfg = {
                            'rank_metric': getattr(config, "OPTIMIZE_MAXIMIZE", "Sharpe Ratio"),
                            'top_percent_pool': getattr(config, "GRID_TOP_POOL", 0.30),
                            'max_candidates': 24,
                            'selection_mode': 'diversified'
                        }
                        
                        # Obtener grid de costes de config
                        cost_grid = getattr(config, "STRESS_COST_GRID", {
                            "commission": [0.0000, 0.0005, 0.0010],
                            "slippage_bps": [0, 5, 10]
                        })
                        
                        # Ejecutar Stress Test - usar el primer DataFrame de df_master como ohlcv
                        stress_df = stress_runner.run(
                            ohlcv=df_master[0],  # Primer DataFrame como OHLCV
                            strategy_cls=strat_cls,
                            grid_df=grid_df,
                            param_cols=param_cols,
                            cost_grid=cost_grid,
                            cfg=stress_cfg
                        )
                        
                        # Generar resumen
                        stress_summary = Stress.summarize(stress_df, how='median')
                        
                        # Guardar resultados Stress Test
                        from backtest import Stress as StressSave
                        StressSave.save(stress_df, strategy_name=f"{base_name}_stress")
                        
                        print(f"[MAIN] Stress Test completado para {base_name}")
                        print(f"[MAIN] Resultados Stress Test guardados para {base_name}")
                        
                except ImportError:
                    print("[MAIN][WARN] Módulo 'overfitting.py' no encontrado, saltando Stress Test")
                except Exception as e:
                    print(f"[MAIN][ERROR] Error en Stress Test para {strat_cls.__name__}: {e}")
                    import traceback
                    traceback.print_exc()

            # ------------------------------------------------------------------
            # 10) (Opcional) Análisis de robustez: Walk-Forward
            # ------------------------------------------------------------------
            if getattr(config, "USE_WF", False):
                print(f"\n[MAIN] Ejecutando Walk-Forward para {strat_cls.__name__}...")
                
                try:
                    from overfitting import WalkForward
                    
                    # Crear instancia de WalkForward con backtest
                    wf_runner = WalkForward(backtest=backtest)
                    
                    # Validar que df_master tenga datos
                    if not df_master or len(df_master) == 0:
                        print(f"[MAIN][WARN] No hay datos en df_master para Walk-Forward, saltando...")
                    elif df_master[0] is None or df_master[0].empty:
                        print(f"[MAIN][WARN] El primer DataFrame de df_master está vacío para Walk-Forward, saltando...")
                    elif grid_df.empty:
                        print(f"[MAIN][WARN] Grid vacío para Walk-Forward, saltando...")
                    else:
                        # Configuración para Walk-Forward
                        wf_cfg = {
                            'wf_config': getattr(config, "WF_CONFIG", {"n_folds": 5, "scheme": "rolling"}),
                            'metric': getattr(config, "OPTIMIZE_MAXIMIZE", "Sharpe Ratio"),
                            'top_percent_pool': getattr(config, "GRID_TOP_POOL", 0.30),
                            'max_candidates': 24,
                            'selection_mode': 'diversified'
                        }
                        
                        # Ejecutar Walk-Forward - usar el primer DataFrame de df_master como ohlcv
                        wf_df = wf_runner.run(
                            ohlcv=df_master[0],  # Primer DataFrame como OHLCV
                            strategy_cls=strat_cls,
                            grid_df=grid_df,
                            param_cols=param_cols,
                            cfg=wf_cfg
                        )
                        
                        # Generar resumen
                        wf_summary = WalkForward.summarize(wf_df)
                        
                        # Guardar resultados Walk-Forward
                        WFSave.save(wf_df, strategy_name=f"{base_name}_wf")
                        
                        print(f"[MAIN] Walk-Forward completado para {base_name}")
                        print(f"[MAIN] Resultados Walk-Forward guardados para {base_name}")
                
                except ImportError:
                    print("[MAIN][WARN] Módulo 'overfitting.py' no encontrado, saltando Walk-Forward")
                except Exception as e:
                    print(f"[MAIN][ERROR] Error en Walk-Forward para {strat_cls.__name__}: {e}")
                    import traceback
                    traceback.print_exc()

            # ----------------------------------------------------------
            # 7c) Guardar heatmaps.html
            # ----------------------------------------------------------
            print(f"\n[MAIN] Generando heatmaps...")
            metric_for_heatmap = getattr(config, "OPTIMIZE_MAXIMIZE", "Sharpe Ratio")
            
            if metric_for_heatmap in grid_df.columns:
                try:
                    Heatmaps.make(
                        grid_df=grid_df,
                        param_cols=param_cols,
                        metric=metric_for_heatmap,
                        strategy_name=f"{base_name}_heatmap"
                    )
                except Exception as e:
                    print(f"[MAIN][ERROR] No se pudo generar heatmap: {e}")
            else:
                print(f"[MAIN][WARN] Métrica {metric_for_heatmap} no en grid_df")
            
            # ----------------------------------------------------------
            # 7d) Seleccionar mejores parámetros
            # ----------------------------------------------------------
            print(f"\n[MAIN] Seleccionando mejores parámetros...")
            
            df_trades, df_metrics, df_indicators = optimizer.select_best_params(
                grid_df=grid_df,
                backtest=backtest,
                strategy_cls=strat_cls,
                df_master=df_master,
                param_cols=param_cols
            )
            
            # Extraer parámetros óptimos
            best_params = {}
            if not grid_df.empty and "__rank__" in grid_df.columns:
                best_idx = grid_df["__rank__"].idxmax()
                best_row = grid_df.loc[best_idx]
                for p in param_cols:
                    if p in best_row.index:
                        best_params[p] = best_row[p]
            
            print(f"[MAIN] Mejores parámetros: {best_params}")
            
            # ----------------------------------------------------------
            # 7e) Guardar stats.xlsx
            # ----------------------------------------------------------
            print(f"\n[MAIN] Guardando stats.xlsx...")
            # Agregar columna con nombre del activo
            df_metrics_copy = df_metrics.copy()
            df_metrics_copy['Asset'] = asset_name

            # Añadir CSCV_PBO si se calculó y USE_CSCV está activo
            if getattr(config, "USE_CSCV", False) and cscv_pbo_value is not None:
                df_metrics_copy['CSCV_PBO'] = cscv_pbo_value
                print(f"[MAIN] CSCV_PBO añadido a stats: {cscv_pbo_value:.3f}")
            elif getattr(config, "USE_CSCV", False):
                # Si USE_CSCV está activo pero no se calculó (por error), añadir NaN
                df_metrics_copy['CSCV_PBO'] = float('nan')
                print(f"[MAIN][WARN] CSCV_PBO no disponible para esta estrategia (error o datos insuficientes)")
            
            Stats.save(df_metrics_copy, strategy_name=f"{base_name}_stats", params=best_params)
            
            # ----------------------------------------------------------
            # 7f) Guardar trades.xlsx
            # ----------------------------------------------------------
            print(f"\n[MAIN] Guardando trades.xlsx...")
            Trades.save(df_trades, strategy_name=f"{base_name}_trades", params=best_params)
            
            # ----------------------------------------------------------
            # 7g) Guardar plots.html
            # ----------------------------------------------------------
            print(f"\n[MAIN] Generando plot interactivo...")
            
            ohlcv = df_master[0] if df_master else None
            
            if ohlcv is not None and not ohlcv.empty:
                try:
                    Plots.make(
                        ohlcv=ohlcv,
                        trades_df=df_trades,
                        indicators_df=df_indicators,
                        strategy_name=f"{base_name}_plot"
                    )
                except Exception as e:
                    print(f"[MAIN][ERROR] No se pudo generar plot: {e}")
            
            print(f"\n[MAIN] {base_name} completado OK")

    # ------------------------------------------------------------------
    # 8) Consolidar results.xlsx
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("[MAIN] Generando resumen consolidado (results.xlsx)...")
    print("="*70)
    
    try:
        summary_df = Results.build_summary(
            order_by=getattr(config, "SUMMARY_FIRST_COL", "Win Rate [%]")
        )
        
        if not summary_df.empty:
            print(f"[MAIN] Resumen generado con {len(summary_df)} resultado(s)")
        else:
            print("[MAIN][WARN] No se generaron resultados para el resumen")
    except Exception as e:
        print(f"[MAIN][ERROR] Error al generar resumen: {e}")
    
        # ------------------------------------------------------------------
    # 9) Re-ejecutar StrategyMultiAsset con parámetros desde results.xlsx
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("[MAIN] Re-ejecutando StrategyMultiAsset con parámetros desde results.xlsx...")
    print("="*70)
    
    # Leer estrategias configuradas desde results.xlsx
    configured_strategies = load_strategies_from_results("results/summary_stats.xlsx")
    
    if not configured_strategies:
        print("[MAIN][INFO] No se encontraron estrategias StrategyMultiAsset en results.xlsx")
    else:
        print(f"[MAIN] Encontradas {len(configured_strategies)} configuraciones desde results.xlsx")
        
        # Para StrategyMultiAsset: ejecutar UNA SOLA VEZ con TODOS los activos de data01
        # Agrupar estrategias por tipo (todas las configuraciones de la misma estrategia)
        strategies_by_type = {}
        # Ejecutar cada tipo de estrategia UNA VEZ con TODOS los activos
        for strat_cls, params_by_asset in configured_strategies:
            print(f"\n[MAIN] Ejecutando {strat_cls.__name__}")
            print(f"[MAIN] Modo multi-activo: TODOS los activos de data01 en un solo cerebro")
            print(f"[MAIN] Parámetros por activo: {params_by_asset}")
            
            # Construir df_master con TODOS los activos de data01 (primera carpeta)
            df_master = []
            if folders_dfs and len(folders_dfs) > 0:
                # Tomar TODOS los activos de la primera carpeta (data01)
                df_master = [df for df in folders_dfs[0] if df is not None and not df.empty]
            
            if not df_master:
                print(f"[MAIN][WARN] No se encontraron activos en data01")
                continue
            
            print(f"[MAIN] df_master construido con {len(df_master)} activo(s) desde data01:")
            for idx, df in enumerate(df_master):
                asset_name = getattr(df, '_asset_name', f'asset_{idx}')
                print(f"  df_master[{idx}]: {asset_name} - {len(df)} filas")
            
            # Configurar parámetros por activo en la clase estrategia
            strat_cls._params_by_asset = params_by_asset
            
            # Ejecutar backtest SIN optimización (solo esta combinación)
            df_tr, df_me, df_ind = backtest.run_backtrader_core(
                df_master=df_master,
                strategy_cls=strat_cls,
                params={},  # Parámetros vacíos, se usarán los específicos por activo
                return_trades=True,
                return_metrics=True,
                return_indicators=True
            )
            
            # Guardar resultados con sufijo especial
            # Usar nombre genérico para multi-asset
            base_name = f"{strat_cls.__name__}_MULTI_ASSET_from_results"
            
            # Definir params como diccionario vacío (los parámetros están en params_by_asset y se aplican por activo)
            params = {}
            
            # Guardar stats (añadir parámetros para guardar también)
            df_metrics_copy = df_me.copy()
            df_metrics_copy['Asset'] = 'MULTI_ASSET_PORTFOLIO'
            Stats.save(df_metrics_copy, strategy_name=f"{base_name}_stats", params=params)
            df_result = df_metrics_copy.copy()
            if params:
                for k, v in params.items():
                    df_result[k] = v
            fname = output_path(f"results/{safe_name(base_name)}.xlsx")
            df_result.to_excel(fname, index=False)
            print(f"[MAIN] Resultados multi-asset guardados: {fname}")
            
            # Guardar trades
            Trades.save(df_tr, strategy_name=f"{base_name}_trades", params=params)
            
            # Guardar plots (usar el primer activo como referencia para OHLCV)
            ohlcv = df_master[0] if df_master else None
            if ohlcv is not None and not ohlcv.empty:
                try:
                    Plots.make(
                        ohlcv=ohlcv,
                        trades_df=df_tr,
                        indicators_df=df_ind,
                        strategy_name=f"{base_name}_plot"
                    )
                except Exception as e:
                    print(f"[MAIN][ERROR] No se pudo generar plot: {e}")
            
            print(f"[MAIN] Resultados guardados para {base_name}")

    # ------------------------------------------------------------------
    # FIN
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("[MAIN] ✓ FLUJO PRINCIPAL COMPLETADO")
    print("="*70)
    print("\nResultados guardados en:")
    print("  - /grid      → Grids de optimización (.xlsx)")
    print("  - /heatmaps  → Heatmaps interactivos (.html)")
    print("  - /stats     → Métricas por estrategia (.xlsx)")
    print("  - /trades    → Trades por estrategia (.xlsx)")
    print("  - /plots     → Gráficos interactivos (.html)")
    print("  - /results   → Resumen consolidado (.xlsx)")
    
    if getattr(config, "USE_CSCV", False):
        print("  - /cscv      → Análisis CSCV (.xlsx)")
    if getattr(config, "USE_STRESS", False):
        print("  - /stress    → Stress tests (.xlsx)")
    if getattr(config, "USE_WF", False):
        print("  - /wf        → Walk-Forward (.xlsx)")
    
    print("\n¡Backtest completado con éxito!\n")


# ====================================================================================
# Punto de entrada
# ====================================================================================
if __name__ == "__main__":
    main()
