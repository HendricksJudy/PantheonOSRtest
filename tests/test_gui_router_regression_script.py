"""Tests for scripts/gui_router_regression.py.

These tests keep the follow-up smoke script stable and make sure it remains
an executable confidence check for the GUI->router path.
"""

import json

import pytest

from scripts import gui_router_regression as script


@pytest.mark.asyncio
async def test_dispatch_check_returns_single_cell_team():
    result = await script.check_auto_dispatch_to_single_cell_team(verbose=False)
    assert result["ok"] is True
    assert result["selected_template"] == "single_cell_team"
    assert result["notice"]["template_id"] == "single_cell_team"


@pytest.mark.asyncio
async def test_plan_validation_rewrites_model_name_tool():
    result = await script.check_router_plan_validation(verbose=False)
    assert result["ok"] is True
    assert result["validate_ok"] is True
    assert result["normalized_tool"] == "scfm_run"
    assert result["normalized_model_name"] == "scgpt"


@pytest.mark.asyncio
async def test_run_returns_zero_when_all_checks_pass(capsys):
    code = await script.run(verbose=False)
    assert code == 0

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["all_passed"] is True
    assert payload["dispatch"]["selected_template"] == "single_cell_team"
    assert payload["validate_plan"]["normalized_tool"] == "scfm_run"
