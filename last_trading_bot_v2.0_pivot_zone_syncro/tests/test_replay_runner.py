"""Tests del runner auxiliar de replay."""
from __future__ import annotations

import sys
import types

import config
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
    assert config.settings.mode == "development"
