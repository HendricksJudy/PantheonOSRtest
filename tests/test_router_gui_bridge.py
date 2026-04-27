"""Tests for the router-to-GUI bridge.

Covers two pieces:
1. UnifiedReviewDialog._normalize_question — defensive coercion of legacy router
   question shapes into the dict layout the dialog renders against.
2. Repl._maybe_promote_router_questions — promotes a router tool result that
   carries `questions` into the same `_pending_approval` slot used by
   notify_user.
"""

from types import SimpleNamespace

from pantheon.repl.core import Repl
from pantheon.repl.viewers.unified_dialog import UnifiedReviewDialog


# =============================================================================
# UnifiedReviewDialog._normalize_question
# =============================================================================


def test_normalize_legacy_string_options():
    q = {
        "field": "batch_key",
        "question": "Which column?",
        "options": ["batch", "sample_id"],
    }
    result = UnifiedReviewDialog._normalize_question(q, fallback_index=0)
    assert result["header"] == "batch_key"
    assert result["input_type"] == "single_choice"
    assert result["options"] == [
        {"value": "batch", "label": "batch", "description": ""},
        {"value": "sample_id", "label": "sample_id", "description": ""},
    ]
    assert result["required"] is True


def test_normalize_missing_input_type_no_options_becomes_text_input():
    q = {"field": "output_path", "question": "Where to save?"}
    result = UnifiedReviewDialog._normalize_question(q, fallback_index=2)
    assert result["input_type"] == "text_input"
    assert result["options"] == []
    assert result["header"] == "output_path"


def test_normalize_already_well_formed_passes_through():
    q = {
        "field": "model",
        "header": "model",
        "question": "Pick one",
        "input_type": "single_choice",
        "options": [{"value": "a", "label": "A", "description": "first"}],
        "required": False,
    }
    result = UnifiedReviewDialog._normalize_question(q, fallback_index=0)
    assert result["header"] == "model"
    assert result["input_type"] == "single_choice"
    assert result["required"] is False
    assert result["options"][0]["description"] == "first"


def test_normalize_falls_back_to_q_index_when_no_field():
    q = {"question": "anything?", "options": []}
    result = UnifiedReviewDialog._normalize_question(q, fallback_index=4)
    assert result["header"] == "Q5"


def test_normalize_partial_option_dicts():
    q = {
        "field": "x",
        "question": "?",
        "input_type": "single_choice",
        "options": [{"value": "v"}, {"label": "L"}],
    }
    result = UnifiedReviewDialog._normalize_question(q, fallback_index=0)
    assert result["options"] == [
        {"value": "v", "label": "v", "description": ""},
        {"value": "L", "label": "L", "description": ""},
    ]


def test_normalize_non_dict_input():
    result = UnifiedReviewDialog._normalize_question("not a dict", fallback_index=0)
    assert result["question"] == ""
    assert result["input_type"] == "text_input"
    assert result["header"] == "Q1"


# =============================================================================
# Repl._maybe_promote_router_questions
# =============================================================================


def _new_repl_stub() -> Repl:
    repl = Repl.__new__(Repl)
    repl._pending_approval = None
    return repl


def test_promote_router_questions_with_raw_dict():
    repl = _new_repl_stub()
    result = {
        "selection": {"recommended": {"name": "scgpt", "rationale": "best fit"}},
        "warnings": ["Data uses symbols but model expects ensembl"],
        "questions": [
            {
                "field": "batch_key",
                "header": "batch_key",
                "question": "Which column?",
                "input_type": "single_choice",
                "options": [
                    {"value": "batch", "label": "batch", "description": ""},
                ],
                "required": True,
            }
        ],
    }
    repl._maybe_promote_router_questions("scfm_router", result, content="")
    assert repl._pending_approval is not None
    assert repl._pending_approval["interrupt"] is True
    assert repl._pending_approval["questions"] == result["questions"]
    assert "scgpt" in repl._pending_approval["message"] or "best fit" in repl._pending_approval["message"]


def test_promote_router_questions_parses_json_string_content():
    repl = _new_repl_stub()
    import json
    payload = {
        "questions": [
            {"field": "model", "question": "Pick a model", "options": ["scgpt", "uce"]}
        ]
    }
    repl._maybe_promote_router_questions(
        "scfm_router", raw_content=None, content=json.dumps(payload)
    )
    assert repl._pending_approval is not None
    assert repl._pending_approval["questions"][0]["field"] == "model"


def test_promote_skips_when_no_questions():
    repl = _new_repl_stub()
    repl._maybe_promote_router_questions(
        "scfm_router", raw_content={"questions": []}, content=""
    )
    assert repl._pending_approval is None


def test_promote_skips_when_question_text_empty():
    repl = _new_repl_stub()
    result = {"questions": [{"field": "x", "question": "  ", "options": []}]}
    repl._maybe_promote_router_questions("scfm_router", result, content="")
    assert repl._pending_approval is None


def test_promote_does_not_clobber_existing_pending_approval():
    repl = _new_repl_stub()
    repl._pending_approval = {"message": "from notify_user", "questions": []}
    result = {"questions": [{"field": "x", "question": "?", "options": ["a"]}]}
    repl._maybe_promote_router_questions("scfm_router", result, content="")
    assert repl._pending_approval == {"message": "from notify_user", "questions": []}


def test_promote_handles_unparseable_content_silently():
    repl = _new_repl_stub()
    repl._maybe_promote_router_questions(
        "scfm_router", raw_content=None, content="not json at all <<<"
    )
    assert repl._pending_approval is None
