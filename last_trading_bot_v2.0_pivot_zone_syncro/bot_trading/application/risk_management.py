"""Gestion de riesgo del bot de trading."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from bot_trading.domain.entities import AccountInfo, RiskLimits, TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class GlobalDrawdownState:
    """Estado persistido del drawdown global."""

    anchor_balance: Optional[float] = None
    anchor_exit_deal_ticket: Optional[int] = None
    anchor_exit_time_utc: Optional[str] = None
    current_drawdown_percent: float = 0.0


@dataclass
class ScopedDrawdownState:
    """Estado persistido del drawdown por simbolo o estrategia."""

    anchor_balance: Optional[float] = None
    anchor_exit_deal_ticket: Optional[int] = None
    anchor_exit_time_utc: Optional[str] = None
    isolated_pnl_since_anchor: float = 0.0
    current_drawdown_percent: float = 0.0


@dataclass
class RiskManager:
    """Evalua limites de riesgo globales, por simbolo y por estrategia."""

    risk_limits: RiskLimits
    global_state: GlobalDrawdownState = field(default_factory=GlobalDrawdownState, init=False)
    symbol_states: dict[str, ScopedDrawdownState] = field(default_factory=dict, init=False)
    strategy_states: dict[str, ScopedDrawdownState] = field(default_factory=dict, init=False)
    _dirty: bool = field(default=False, init=False, repr=False)
    _state_path: Optional[Path] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._state_path = self._resolve_state_path(self.risk_limits.risk_state_path)
        self._load_state()

    @staticmethod
    def _resolve_state_path(path_value: Optional[str]) -> Optional[Path]:
        if not path_value:
            return None
        path = Path(path_value)
        if path.is_absolute():
            return path
        repo_root = Path(__file__).resolve().parents[2]
        return (repo_root / path).resolve()

    def _load_state(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return

        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            self.global_state = self._parse_global_state(payload.get("global"))
            self.symbol_states = self._parse_scoped_states(payload.get("symbols"))
            self.strategy_states = self._parse_scoped_states(payload.get("strategies"))
            logger.info("Estado de riesgo cargado desde %s", self._state_path)
        except Exception as exc:
            logger.warning("No se pudo cargar estado de riesgo (%s): %s", self._state_path, exc)

    def _save_state_if_dirty(self) -> None:
        if not self._dirty or self._state_path is None:
            return

        payload = {
            "global": asdict(self.global_state),
            "symbols": {key: asdict(value) for key, value in sorted(self.symbol_states.items())},
            "strategies": {key: asdict(value) for key, value in sorted(self.strategy_states.items())},
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            self._dirty = False
        except Exception as exc:
            logger.warning("No se pudo persistir estado de riesgo (%s): %s", self._state_path, exc)

    @staticmethod
    def _parse_global_state(raw: object) -> GlobalDrawdownState:
        if not isinstance(raw, dict):
            return GlobalDrawdownState()
        return GlobalDrawdownState(
            anchor_balance=RiskManager._to_optional_float(raw.get("anchor_balance")),
            anchor_exit_deal_ticket=RiskManager._to_optional_int(raw.get("anchor_exit_deal_ticket")),
            anchor_exit_time_utc=RiskManager._to_optional_str(raw.get("anchor_exit_time_utc")),
            current_drawdown_percent=RiskManager._to_float(raw.get("current_drawdown_percent"), 0.0),
        )

    @staticmethod
    def _parse_scoped_states(raw: object) -> dict[str, ScopedDrawdownState]:
        if not isinstance(raw, dict):
            return {}
        result: dict[str, ScopedDrawdownState] = {}
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            result[str(key)] = ScopedDrawdownState(
                anchor_balance=RiskManager._to_optional_float(value.get("anchor_balance")),
                anchor_exit_deal_ticket=RiskManager._to_optional_int(value.get("anchor_exit_deal_ticket")),
                anchor_exit_time_utc=RiskManager._to_optional_str(value.get("anchor_exit_time_utc")),
                isolated_pnl_since_anchor=RiskManager._to_float(value.get("isolated_pnl_since_anchor"), 0.0),
                current_drawdown_percent=RiskManager._to_float(value.get("current_drawdown_percent"), 0.0),
            )
        return result

    @staticmethod
    def _to_optional_float(value: object) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_float(value: object, default: float = 0.0) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _to_optional_int(value: object) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_optional_str(value: object) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @staticmethod
    def _trade_sort_key(trade: TradeRecord) -> tuple[datetime, int]:
        exit_ticket = int(trade.exit_deal_ticket or 0)
        return (trade.exit_time, exit_ticket)

    @staticmethod
    def _drawdown_from_balances(anchor_balance: Optional[float], current_balance: Optional[float]) -> float:
        if anchor_balance is None or current_balance is None or anchor_balance <= 0:
            return 0.0
        loss = anchor_balance - current_balance
        if loss <= 0:
            return 0.0
        return (loss / anchor_balance) * 100.0

    def has_drawdown_limits(self) -> bool:
        """Indica si existe algun limite de drawdown configurado."""
        return (
            self.risk_limits.dd_global is not None
            or bool(self.risk_limits.dd_por_activo)
            or bool(self.risk_limits.dd_por_estrategia)
        )

    def apply_closed_trades(self, new_trades: list[TradeRecord], current_balance: float) -> None:
        """Aplica trades cerrados nuevos y recalcula el drawdown actual."""
        ordered_trades = sorted(new_trades, key=self._trade_sort_key)
        balance_after_trade = self._reconstruct_balance_after_trade(ordered_trades, current_balance)

        for trade in ordered_trades:
            trade_balance_after = balance_after_trade[self._trade_identity(trade)]
            self._apply_global_trade(trade, trade_balance_after)
            self._apply_symbol_trade(trade, trade_balance_after)
            self._apply_strategy_trade(trade, trade_balance_after)

        self._recalculate_current_drawdowns(current_balance)
        self._save_state_if_dirty()

    def _reconstruct_balance_after_trade(
        self,
        ordered_trades: list[TradeRecord],
        current_balance: float,
    ) -> dict[tuple, float]:
        result: dict[tuple, float] = {}
        running_balance = float(current_balance)
        for trade in reversed(ordered_trades):
            result[self._trade_identity(trade)] = running_balance
            running_balance -= float(trade.pnl)
        return result

    @staticmethod
    def _trade_identity(trade: TradeRecord) -> tuple:
        if trade.position_id is not None and trade.exit_deal_ticket is not None:
            return ("position_exit_deal", int(trade.position_id), int(trade.exit_deal_ticket))
        if trade.exit_deal_ticket is not None:
            return ("exit_deal", int(trade.exit_deal_ticket))
        return (
            "fallback",
            trade.symbol,
            trade.strategy_name,
            trade.exit_time.isoformat(),
            float(trade.pnl),
        )

    def _apply_global_trade(self, trade: TradeRecord, balance_after_trade: float) -> None:
        if trade.pnl <= 0:
            return
        self._set_global_anchor(
            balance_after_trade=balance_after_trade,
            exit_deal_ticket=trade.exit_deal_ticket,
            exit_time_utc=trade.exit_time.isoformat(),
        )

    def _apply_symbol_trade(self, trade: TradeRecord, balance_after_trade: float) -> None:
        state = self.symbol_states.setdefault(trade.symbol, ScopedDrawdownState())
        self._apply_scoped_trade(
            state=state,
            trade=trade,
            balance_after_trade=balance_after_trade,
        )

    def _apply_strategy_trade(self, trade: TradeRecord, balance_after_trade: float) -> None:
        state = self.strategy_states.setdefault(trade.strategy_name, ScopedDrawdownState())
        self._apply_scoped_trade(
            state=state,
            trade=trade,
            balance_after_trade=balance_after_trade,
        )

    def _apply_scoped_trade(
        self,
        *,
        state: ScopedDrawdownState,
        trade: TradeRecord,
        balance_after_trade: float,
    ) -> None:
        if trade.pnl > 0:
            self._set_scoped_anchor(
                state,
                balance_after_trade=balance_after_trade,
                exit_deal_ticket=trade.exit_deal_ticket,
                exit_time_utc=trade.exit_time.isoformat(),
            )
            return

        if state.anchor_balance is None:
            return

        self._set_if_changed(state, "isolated_pnl_since_anchor", state.isolated_pnl_since_anchor + float(trade.pnl))
        synthetic_balance = state.anchor_balance + state.isolated_pnl_since_anchor
        self._set_if_changed(
            state,
            "current_drawdown_percent",
            self._drawdown_from_balances(state.anchor_balance, synthetic_balance),
        )

    def _set_global_anchor(
        self,
        *,
        balance_after_trade: float,
        exit_deal_ticket: Optional[int],
        exit_time_utc: str,
    ) -> None:
        self._set_if_changed(self.global_state, "anchor_balance", float(balance_after_trade))
        self._set_if_changed(self.global_state, "anchor_exit_deal_ticket", exit_deal_ticket)
        self._set_if_changed(self.global_state, "anchor_exit_time_utc", exit_time_utc)
        self._set_if_changed(self.global_state, "current_drawdown_percent", 0.0)

    def _set_scoped_anchor(
        self,
        state: ScopedDrawdownState,
        *,
        balance_after_trade: float,
        exit_deal_ticket: Optional[int],
        exit_time_utc: str,
    ) -> None:
        self._set_if_changed(state, "anchor_balance", float(balance_after_trade))
        self._set_if_changed(state, "anchor_exit_deal_ticket", exit_deal_ticket)
        self._set_if_changed(state, "anchor_exit_time_utc", exit_time_utc)
        self._set_if_changed(state, "isolated_pnl_since_anchor", 0.0)
        self._set_if_changed(state, "current_drawdown_percent", 0.0)

    def _recalculate_current_drawdowns(self, current_balance: float) -> None:
        self._set_if_changed(
            self.global_state,
            "current_drawdown_percent",
            self._drawdown_from_balances(self.global_state.anchor_balance, float(current_balance)),
        )

        for state in self.symbol_states.values():
            synthetic_balance = (
                None if state.anchor_balance is None else state.anchor_balance + state.isolated_pnl_since_anchor
            )
            self._set_if_changed(
                state,
                "current_drawdown_percent",
                self._drawdown_from_balances(state.anchor_balance, synthetic_balance),
            )

        for state in self.strategy_states.values():
            synthetic_balance = (
                None if state.anchor_balance is None else state.anchor_balance + state.isolated_pnl_since_anchor
            )
            self._set_if_changed(
                state,
                "current_drawdown_percent",
                self._drawdown_from_balances(state.anchor_balance, synthetic_balance),
            )

    def _set_if_changed(self, target: object, attr: str, value: object) -> None:
        if getattr(target, attr) == value:
            return
        setattr(target, attr, value)
        self._dirty = True

    def check_bot_risk_limits(self, current_balance: float) -> bool:
        """Valida si el bot puede operar segun el drawdown global actual."""
        if self.risk_limits.dd_global is None:
            return True
        drawdown = self.global_state.current_drawdown_percent
        allowed = drawdown <= self.risk_limits.dd_global
        if not allowed:
            anchor_balance = self.global_state.anchor_balance
            loss = 0.0 if anchor_balance is None else max(0.0, anchor_balance - float(current_balance))
            logger.warning(
                "Limite drawdown global superado scope=global drawdown=%.2f%% limit=%.2f%% anchor_balance=%.2f current_balance=%.2f loss_since_anchor=%.2f anchor_exit_deal_ticket=%s anchor_exit_time_utc=%s",
                drawdown,
                self.risk_limits.dd_global,
                anchor_balance or 0.0,
                float(current_balance),
                loss,
                self.global_state.anchor_exit_deal_ticket,
                self.global_state.anchor_exit_time_utc,
            )
        return allowed

    def check_symbol_risk_limits(self, symbol: str) -> bool:
        """Valida limites de riesgo por simbolo."""
        limit = self.risk_limits.dd_por_activo.get(symbol)
        if limit is None:
            return True
        state = self.symbol_states.get(symbol)
        drawdown = state.current_drawdown_percent if state else 0.0
        allowed = drawdown <= limit
        if not allowed and state is not None:
            synthetic_balance = (state.anchor_balance or 0.0) + state.isolated_pnl_since_anchor
            loss = max(0.0, (state.anchor_balance or 0.0) - synthetic_balance)
            logger.warning(
                "Limite drawdown simbolo superado scope=symbol symbol=%s drawdown=%.2f%% limit=%.2f%% anchor_balance=%.2f synthetic_balance=%.2f loss_since_anchor=%.2f anchor_exit_deal_ticket=%s anchor_exit_time_utc=%s",
                symbol,
                drawdown,
                limit,
                state.anchor_balance or 0.0,
                synthetic_balance,
                loss,
                state.anchor_exit_deal_ticket,
                state.anchor_exit_time_utc,
            )
        return allowed

    def check_strategy_risk_limits(self, strategy_name: str) -> bool:
        """Valida limites de riesgo por estrategia."""
        limit = self.risk_limits.dd_por_estrategia.get(strategy_name)
        if limit is None:
            return True
        state = self.strategy_states.get(strategy_name)
        drawdown = state.current_drawdown_percent if state else 0.0
        allowed = drawdown <= limit
        if not allowed and state is not None:
            synthetic_balance = (state.anchor_balance or 0.0) + state.isolated_pnl_since_anchor
            loss = max(0.0, (state.anchor_balance or 0.0) - synthetic_balance)
            logger.warning(
                "Limite drawdown estrategia superado scope=strategy strategy=%s drawdown=%.2f%% limit=%.2f%% anchor_balance=%.2f synthetic_balance=%.2f loss_since_anchor=%.2f anchor_exit_deal_ticket=%s anchor_exit_time_utc=%s",
                strategy_name,
                drawdown,
                limit,
                state.anchor_balance or 0.0,
                synthetic_balance,
                loss,
                state.anchor_exit_deal_ticket,
                state.anchor_exit_time_utc,
            )
        return allowed

    def check_margin_limits(
        self,
        account_info: AccountInfo,
        required_margin: Optional[float] = None,
    ) -> bool:
        """Valida limites de margen antes de abrir nuevas posiciones.

        Verifica dos condiciones:
        1. Si el margen usado supera el porcentaje maximo configurado.
        2. Si hay margen libre suficiente para la orden solicitada.

        Args:
            account_info: Informacion de cuenta del broker.
            required_margin: Margen requerido para la nueva orden (opcional).

        Returns:
            True si se puede abrir la posicion, False si se supera algun limite.
        """
        max_margin_percent = self.risk_limits.max_margin_usage_percent
        if max_margin_percent is None:
            logger.debug("No hay limite de margen configurado, permitiendo orden")
            return True

        if account_info.equity <= 0:
            logger.warning("Equity es 0 o negativo, bloqueando orden por seguridad")
            return False

        margin_usage_percent = (account_info.margin / account_info.equity) * 100

        if margin_usage_percent > max_margin_percent:
            logger.warning(
                "Limite de margen superado: %.2f%% > %.2f%% (Margin: %.2f, Equity: %.2f)",
                margin_usage_percent,
                max_margin_percent,
                account_info.margin,
                account_info.equity,
            )
            return False

        if required_margin is not None and required_margin > 0:
            if account_info.margin_free < required_margin:
                logger.warning(
                    "Margen libre insuficiente: %.2f < %.2f requerido (MarginFree: %.2f)",
                    account_info.margin_free,
                    required_margin,
                    account_info.margin_free,
                )
                return False

            total_margin = account_info.margin + required_margin
            total_margin_percent = (total_margin / account_info.equity) * 100

            if total_margin_percent > max_margin_percent:
                logger.warning(
                    "Abrir esta posicion superaria el limite de margen: %.2f%% > %.2f%% (Margin actual: %.2f, Requerido: %.2f, Total: %.2f)",
                    total_margin_percent,
                    max_margin_percent,
                    account_info.margin,
                    required_margin,
                    total_margin,
                )
                return False

        logger.debug(
            "Validacion de margen exitosa: %.2f%% usado (limite: %.2f%%), MarginFree: %.2f",
            margin_usage_percent,
            max_margin_percent,
            account_info.margin_free,
        )
        return True
