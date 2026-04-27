"""Unit tests for the SCFMToolSet.scfm_validate_plan @tool."""

import asyncio
import json

import pytest

from pantheon.toolsets.scfm.toolset import SCFMToolSet


def _valid_plan_dict() -> dict:
    return {
        "intent": {"task": "embed", "confidence": 0.9, "constraints": {}},
        "inputs": {"query": "Embed with scgpt", "adata_path": None},
        "data_profile": None,
        "selection": {
            "recommended": {"name": "scgpt", "rationale": "general embed"},
            "fallbacks": [],
        },
        "resolved_params": {},
        "plan": [{"tool": "scfm_run", "args": {"model_name": "scgpt", "task": "embed"}}],
        "questions": [],
        "warnings": [],
    }


@pytest.fixture
def toolset():
    return SCFMToolSet()


def _run(coro):
    return asyncio.run(coro)


class TestScfmValidatePlan:
    def test_happy_path_dict(self, toolset):
        r = _run(toolset.scfm_validate_plan(_valid_plan_dict()))
        assert r["ok"] is True
        assert r["errors"] == []
        assert r["normalized_plan"]["plan"][0]["tool"] == "scfm_run"

    def test_happy_path_json_string(self, toolset):
        r = _run(toolset.scfm_validate_plan(json.dumps(_valid_plan_dict())))
        assert r["ok"] is True
        assert r["normalized_plan"]["selection"]["recommended"]["name"] == "scgpt"

    def test_model_name_as_tool_is_normalized(self, toolset):
        plan = _valid_plan_dict()
        plan["plan"] = [{"tool": "scgpt", "args": {"adata_path": "x.h5ad"}}]
        r = _run(toolset.scfm_validate_plan(plan))
        assert r["ok"] is True
        step = r["normalized_plan"]["plan"][0]
        assert step["tool"] == "scfm_run"
        assert step["args"]["model_name"] == "scgpt"
        assert step["args"]["adata_path"] == "x.h5ad"

    def test_malformed_json_string(self, toolset):
        r = _run(toolset.scfm_validate_plan("not actually json"))
        assert r["ok"] is False
        assert r["normalized_plan"] is None
        assert any("not valid JSON" in e for e in r["errors"])

    def test_unknown_model_in_selection(self, toolset):
        plan = _valid_plan_dict()
        plan["selection"]["recommended"]["name"] = "not_a_real_model"
        r = _run(toolset.scfm_validate_plan(plan))
        assert r["ok"] is False
        assert any("not found in registry" in e for e in r["errors"])

    def test_invalid_tool_that_is_not_a_known_model(self, toolset):
        plan = _valid_plan_dict()
        plan["plan"] = [{"tool": "totally_made_up", "args": {}}]
        r = _run(toolset.scfm_validate_plan(plan))
        assert r["ok"] is False
        # Either surfaced by Pydantic's tool validator or by the registry check.
        assert any("tool" in e.lower() for e in r["errors"])

    def test_wrong_type_for_plan_argument(self, toolset):
        r = _run(toolset.scfm_validate_plan(12345))
        assert r["ok"] is False
        assert r["normalized_plan"] is None
        assert any("JSON string or dict" in e for e in r["errors"])
