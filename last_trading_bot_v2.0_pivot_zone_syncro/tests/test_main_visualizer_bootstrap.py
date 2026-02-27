"""Tests de bootstrap del autovisualizador en el entrypoint principal."""
from __future__ import annotations

from pathlib import Path

import bot_trading.main as main_mod
from bot_trading.domain.entities import SymbolConfig


def test_build_auto_visualizer_habilitado_por_defecto(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VISUALIZER_AUTO_ENABLE", raising=False)
    monkeypatch.setenv("VISUALIZER_REFRESH_SECONDS", "17")
    monkeypatch.setattr(main_mod, "ROOT", tmp_path)

    captured: dict = {}

    class DummyAutoVisualizerService:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(main_mod, "AutoVisualizerService", DummyAutoVisualizerService)

    result = main_mod._build_auto_visualizer([SymbolConfig(name="EURUSD", min_timeframe="M1")])

    assert isinstance(result, DummyAutoVisualizerService)
    assert captured["refresh_seconds"] == 17
    assert captured["start_from_end"] is True
    assert captured["symbols"][0].name == "EURUSD"


def test_build_auto_visualizer_se_puede_desactivar_por_env(monkeypatch) -> None:
    monkeypatch.setenv("VISUALIZER_AUTO_ENABLE", "0")

    result = main_mod._build_auto_visualizer([SymbolConfig(name="EURUSD", min_timeframe="M1")])

    assert result is None
