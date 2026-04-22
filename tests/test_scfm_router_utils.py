"""
Unit tests for the surviving scFM router utilities in `pantheon.toolsets.scfm.router`.

After the deprecated `scfm_router` tool and its orchestration helpers were removed,
only the data models (Pydantic), `_normalize_router_output_dict`,
`validate_router_output`, and `build_model_cards` remain. These are used by the
`scfm_validate_plan` tool (see `tests/test_scfm_validate_plan.py`) and as a reference
schema for the `fm_router` sub-agent template.
"""

import pytest

from pantheon.toolsets.scfm.router import (
    VALID_TASKS,
    VALID_SCFM_TOOLS,
    ModelSelection,
    Question,
    ResolvedParams,
    RouterInputs,
    RouterIntent,
    RouterOutput,
    RouterSelection,
    ToolCall,
    _normalize_router_output_dict,
    build_model_cards,
    validate_router_output,
)


class TestRouterIntent:
    def test_valid_task(self):
        for task in VALID_TASKS:
            assert RouterIntent(task=task, confidence=0.9).task == task

    def test_invalid_task(self):
        with pytest.raises(ValueError):
            RouterIntent(task="invalid_task", confidence=0.9)

    def test_confidence_bounds(self):
        assert RouterIntent(task="embed", confidence=0.5).confidence == 0.5
        with pytest.raises(ValueError):
            RouterIntent(task="embed", confidence=1.5)
        with pytest.raises(ValueError):
            RouterIntent(task="embed", confidence=-0.1)


class TestToolCall:
    def test_valid_tools(self):
        for name in VALID_SCFM_TOOLS:
            assert ToolCall(tool=name, args={}).tool == name

    def test_invalid_tool(self):
        with pytest.raises(ValueError):
            ToolCall(tool="invalid_tool", args={})


class TestRouterOutput:
    def test_minimal_valid_output(self):
        output = RouterOutput(
            intent=RouterIntent(task="embed", confidence=0.9),
            inputs=RouterInputs(query="Embed my data"),
            selection=RouterSelection(
                recommended=ModelSelection(name="uce", rationale="Good model")
            ),
        )
        assert output.intent.task == "embed"
        assert output.selection.recommended.name == "uce"

    def test_full_output(self):
        output = RouterOutput(
            intent=RouterIntent(task="integrate", confidence=0.85),
            inputs=RouterInputs(query="Integrate batches", adata_path="/data/test.h5ad"),
            data_profile={"n_cells": 1000, "species": "human"},
            selection=RouterSelection(
                recommended=ModelSelection(name="scgpt", rationale="Best for integration"),
                fallbacks=[ModelSelection(name="uce", rationale="Alternative")],
            ),
            resolved_params=ResolvedParams(output_path="/data/output.h5ad", batch_key="batch_id"),
            plan=[
                ToolCall(tool="scfm_preprocess_validate", args={}),
                ToolCall(tool="scfm_run", args={}),
            ],
            questions=[Question(field="batch_key", question="Which column?", options=["batch"])],
            warnings=["Data may need preprocessing"],
        )
        assert output.intent.task == "integrate"
        assert len(output.plan) == 2
        assert len(output.questions) == 1


class TestValidateRouterOutput:
    def test_valid_output_passes(self):
        valid_output = {
            "intent": {"task": "embed", "confidence": 0.9, "constraints": {}},
            "inputs": {"query": "Embed my data", "adata_path": None},
            "data_profile": None,
            "selection": {
                "recommended": {"name": "uce", "rationale": "Good model"},
                "fallbacks": [],
            },
            "resolved_params": {},
            "plan": [],
            "questions": [],
            "warnings": [],
        }
        is_valid, errors, parsed = validate_router_output(valid_output)
        assert is_valid
        assert errors == []
        assert parsed is not None

    def test_invalid_task_fails(self):
        invalid_output = {
            "intent": {"task": "invalid_task", "confidence": 0.9, "constraints": {}},
            "inputs": {"query": "Embed my data"},
            "selection": {
                "recommended": {"name": "uce", "rationale": ""},
                "fallbacks": [],
            },
        }
        is_valid, errors, _ = validate_router_output(invalid_output)
        assert not is_valid
        assert any("task" in e.lower() for e in errors)

    def test_unknown_model_fails(self):
        invalid_output = {
            "intent": {"task": "embed", "confidence": 0.9, "constraints": {}},
            "inputs": {"query": "Embed my data"},
            "selection": {
                "recommended": {"name": "nonexistent_model", "rationale": ""},
                "fallbacks": [],
            },
        }
        is_valid, errors, _ = validate_router_output(invalid_output)
        assert not is_valid
        assert any("not found in registry" in e for e in errors)

    def test_invalid_tool_fails(self):
        invalid_output = {
            "intent": {"task": "embed", "confidence": 0.9, "constraints": {}},
            "inputs": {"query": "Embed my data"},
            "selection": {
                "recommended": {"name": "uce", "rationale": ""},
                "fallbacks": [],
            },
            "plan": [{"tool": "totally_not_a_tool", "args": {}}],
        }
        is_valid, errors, _ = validate_router_output(invalid_output)
        assert not is_valid
        assert any("Invalid tool" in e or "tool" in e.lower() for e in errors)

    def test_model_name_in_plan_is_normalized_to_scfm_run(self):
        output = {
            "intent": {"task": "embed", "confidence": 0.9, "constraints": {}},
            "inputs": {"query": "Embed plant data"},
            "selection": {
                "recommended": {"name": "scplantllm", "rationale": "plant model"},
                "fallbacks": [],
            },
            "plan": [{"tool": "scplantllm", "args": {"adata_path": "plant.h5ad"}}],
        }
        is_valid, errors, parsed = validate_router_output(output)
        assert is_valid, errors
        assert parsed is not None
        assert parsed.plan[0].tool == "scfm_run"
        assert parsed.plan[0].args["model_name"] == "scplantllm"


class TestNormalizeRouterOutputDict:
    def test_non_dict_returns_as_is(self):
        assert _normalize_router_output_dict("not a dict") == "not a dict"

    def test_plan_not_a_list_is_untouched(self):
        out = _normalize_router_output_dict({"plan": "oops"})
        assert out == {"plan": "oops"}

    def test_known_model_name_rewritten_to_scfm_run(self):
        out = _normalize_router_output_dict(
            {"plan": [{"tool": "scgpt", "args": {"adata_path": "x.h5ad"}}]}
        )
        step = out["plan"][0]
        assert step["tool"] == "scfm_run"
        assert step["args"]["model_name"] == "scgpt"
        assert step["args"]["adata_path"] == "x.h5ad"

    def test_unknown_tool_left_untouched(self):
        out = _normalize_router_output_dict(
            {"plan": [{"tool": "totally_unknown", "args": {}}]}
        )
        # Neither a valid scfm tool nor a registered model, so stays as-is.
        assert out["plan"][0]["tool"] == "totally_unknown"

    def test_existing_valid_tool_untouched(self):
        out = _normalize_router_output_dict(
            {"plan": [{"tool": "scfm_run", "args": {"model_name": "scgpt"}}]}
        )
        assert out["plan"][0]["tool"] == "scfm_run"
        assert out["plan"][0]["args"]["model_name"] == "scgpt"


class TestBuildModelCards:
    def test_builds_cards(self):
        cards = build_model_cards()
        lowered = cards.lower()
        assert "uce" in lowered
        assert "scgpt" in lowered
        assert "geneformer" in lowered

    def test_skill_ready_filter_is_subset(self):
        all_cards = build_model_cards(skill_ready_only=False)
        ready_cards = build_model_cards(skill_ready_only=True)
        assert len(ready_cards) <= len(all_cards)
