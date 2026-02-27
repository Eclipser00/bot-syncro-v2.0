"""Utilidades para exportar información a archivos Excel."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from bot_trading.domain.entities import TradeRecord

logger = logging.getLogger(__name__)


class ExcelExporter:
    """Exporta registros de trades y métricas a archivos .xlsx."""

    def export_trades(self, trades: Iterable[TradeRecord], file_path: Path) -> None:
        """Exporta una colección de TradeRecord a un archivo Excel.

        Args:
            trades: Registros a exportar.
            file_path: Ruta del archivo destino.
        """
        df = pd.DataFrame([trade.__dict__ for trade in trades])
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Exportando %d trades a %s", len(df), file_path)
        df.to_excel(file_path, index=False)
