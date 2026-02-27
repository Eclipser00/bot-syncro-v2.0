"""Gestión de riesgo del bot de trading."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from bot_trading.domain.entities import AccountInfo, RiskLimits, TradeRecord

logger = logging.getLogger(__name__)


class RiskLimitExceeded(Exception):
    """Excepción para indicar que un límite de riesgo ha sido violado."""


@dataclass
class RiskManager:
    """Evalúa límites de riesgo globales, por símbolo y por estrategia."""

    risk_limits: RiskLimits

    def _calculate_drawdown(self, trades: list[TradeRecord]) -> float:
        """Calcula el drawdown real como porcentaje desde el máximo histórico.
        
        Usa el balance inicial configurado para calcular correctamente el drawdown
        incluso cuando la estrategia comienza con pérdidas.
        
        Args:
            trades: Lista de trades cerrados ordenados cronológicamente.
            
        Returns:
            Drawdown actual en porcentaje (0-100).
        """
        if not trades:
            return 0.0
        
        # Usar balance inicial de la configuración
        initial_balance = self.risk_limits.initial_balance
        equity = initial_balance
        max_equity = initial_balance
        max_drawdown = 0.0
        
        for trade in trades:
            equity += trade.pnl
            if equity > max_equity:
                max_equity = equity
            
            # Calcular drawdown actual (max_equity siempre > 0 con balance inicial)
            current_dd = ((max_equity - equity) / max_equity) * 100
            if current_dd > max_drawdown:
                max_drawdown = current_dd
        
        logger.debug("Drawdown calculado: %.2f%% (Equity: %.2f, Max: %.2f)", 
                     max_drawdown, equity, max_equity)
        return max_drawdown

    def check_bot_risk_limits(self, trades: list[TradeRecord]) -> bool:
        """Valida si el bot puede operar según el drawdown global."""
        if self.risk_limits.dd_global is None:
            return True
        drawdown = self._calculate_drawdown(trades)
        allowed = drawdown <= self.risk_limits.dd_global
        if not allowed:
            logger.warning(
                "Límite de drawdown global superado: %.2f%% > %.2f%%",
                drawdown,
                self.risk_limits.dd_global,
            )
        return allowed

    def check_symbol_risk_limits(
        self, symbol: str, trades: list[TradeRecord]
    ) -> bool:
        """Valida límites de riesgo por símbolo."""
        limit = self.risk_limits.dd_por_activo.get(symbol)
        if limit is None:
            return True
        filtered = [t for t in trades if t.symbol == symbol]
        drawdown = self._calculate_drawdown(filtered)
        allowed = drawdown <= limit
        if not allowed:
            logger.warning(
                "Límite de drawdown por símbolo %s superado: %.2f%% > %.2f%%",
                symbol,
                drawdown,
                limit,
            )
        return allowed

    def check_strategy_risk_limits(
        self, strategy_name: str, trades: list[TradeRecord]
    ) -> bool:
        """Valida límites de riesgo por estrategia."""
        limit = self.risk_limits.dd_por_estrategia.get(strategy_name)
        if limit is None:
            return True
        filtered = [t for t in trades if t.strategy_name == strategy_name]
        drawdown = self._calculate_drawdown(filtered)
        allowed = drawdown <= limit
        if not allowed:
            logger.warning(
                "Límite de drawdown por estrategia %s superado: %.2f%% > %.2f%%",
                strategy_name,
                drawdown,
                limit,
            )
        return allowed

    def check_margin_limits(
        self, 
        account_info: AccountInfo, 
        required_margin: Optional[float] = None
    ) -> bool:
        """Valida límites de margen antes de abrir nuevas posiciones.
        
        Verifica dos condiciones:
        1. Si el margen usado supera el porcentaje máximo configurado.
        2. Si hay margen libre suficiente para la orden solicitada.
        
        Args:
            account_info: Información de cuenta del broker.
            required_margin: Margen requerido para la nueva orden (opcional).
            
        Returns:
            True si se puede abrir la posición, False si se supera algún límite.
        """
        # Si no hay límite configurado, permitir todas las órdenes
        max_margin_percent = self.risk_limits.max_margin_usage_percent
        if max_margin_percent is None:
            logger.debug("No hay límite de margen configurado, permitiendo orden")
            return True
        
        # Calcular porcentaje de margen usado actual
        if account_info.equity <= 0:
            logger.warning("Equity es 0 o negativo, bloqueando orden por seguridad")
            return False
        
        margin_usage_percent = (account_info.margin / account_info.equity) * 100
        
        # Verificar si el margen usado supera el límite
        if margin_usage_percent > max_margin_percent:
            logger.warning(
                "Límite de margen superado: %.2f%% > %.2f%% (Margin: %.2f, Equity: %.2f)",
                margin_usage_percent,
                max_margin_percent,
                account_info.margin,
                account_info.equity,
            )
            return False
        
        # Si se especifica margen requerido, verificar que hay suficiente margen libre
        if required_margin is not None and required_margin > 0:
            if account_info.margin_free < required_margin:
                logger.warning(
                    "Margen libre insuficiente: %.2f < %.2f requerido (MarginFree: %.2f)",
                    account_info.margin_free,
                    required_margin,
                    account_info.margin_free,
                )
                return False
            
            # Verificar que el margen total (usado + requerido) no supere el límite
            total_margin = account_info.margin + required_margin
            total_margin_percent = (total_margin / account_info.equity) * 100
            
            if total_margin_percent > max_margin_percent:
                logger.warning(
                    "Abrir esta posición superaría el límite de margen: %.2f%% > %.2f%% "
                    "(Margin actual: %.2f, Requerido: %.2f, Total: %.2f)",
                    total_margin_percent,
                    max_margin_percent,
                    account_info.margin,
                    required_margin,
                    total_margin,
                )
                return False
        
        logger.debug(
            "Validación de margen exitosa: %.2f%% usado (límite: %.2f%%), "
            "MarginFree: %.2f",
            margin_usage_percent,
            max_margin_percent,
            account_info.margin_free,
        )
        return True
