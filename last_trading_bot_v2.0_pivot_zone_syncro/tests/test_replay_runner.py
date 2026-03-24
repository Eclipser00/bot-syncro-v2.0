"""Tests del runner auxiliar de replay."""
from __future__ import annotations

import sys
import types

import config
import bot_trading.main as main_mod
from bot_trading import replay_runner


def test_replay_runner_fuerza_modo_development(monkeypatch) -> None:
    original_env = config.ACTIVE_ENV
    original_settings = config.settings

    captured: dict[str, str] = {}
    fake_main_module = types.SimpleNamespace(main=lambda: captured.setdefault("mode", config.settings.mode))

    monkeypatch.setattr(config, "ACTIVE_ENV", "production")
    config.settings = config.load_settings("production")
    monkeypatch.setitem(sys.modules, "bot_trading.main", fake_main_module)

    try:
        replay_runner.run()
    finally:
        config.ACTIVE_ENV = original_env
        config.settings = original_settings

    assert captured["mode"] == "development"
    assert config.settings.mode == original_settings.mode


def test_load_settings_expone_risk_state_path_en_todos_los_entornos() -> None:
    for env in ("production", "development", "testing"):
        settings = config.load_settings(env)
        assert settings.risk.risk_state_path == "outputs/risk_state.json"


def test_build_risk_manager_no_falla_con_production(monkeypatch) -> None:
    production_settings = config.load_settings("production")
    monkeypatch.setattr(main_mod, "settings", production_settings)

    risk_manager = main_mod._build_risk_manager()

    assert risk_manager.risk_limits.risk_state_path == "outputs/risk_state.json"
