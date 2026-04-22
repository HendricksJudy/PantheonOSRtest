"""Unit tests for pantheon.chatroom.special_agents.TeamDispatcher."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pantheon.chatroom.special_agents import TeamDispatcher, TeamSelection


AVAILABLE = [
    {
        "id": "default",
        "name": "Default Assistant",
        "description": "Generic helper",
        "category": "general",
    },
    {
        "id": "single_cell_team",
        "name": "Single Cell Analysis Team",
        "description": "scFM routing and single-cell analysis",
        "category": "bioinformatics",
    },
    {
        "id": "omicverse_team",
        "name": "Omicverse Team",
        "description": "Omics analysis pipelines",
        "category": "bioinformatics",
    },
]


def _run(coro):
    return asyncio.run(coro)


def _mock_agent(response_content: str | Exception):
    """Build a MagicMock Agent whose .run() returns an object with .content."""
    agent = MagicMock()
    if isinstance(response_content, Exception):
        agent.run = AsyncMock(side_effect=response_content)
    else:
        agent.run = AsyncMock(return_value=SimpleNamespace(content=response_content))
    return agent


class TestTeamDispatcher:
    def test_picks_valid_known_id(self):
        dispatcher = TeamDispatcher()
        dispatcher._dispatch_agent = _mock_agent(
            json.dumps({
                "template_id": "single_cell_team",
                "reason": "mentions scGPT",
                "confidence": 0.9,
            })
        )
        dispatcher._dispatch_agent_model = "stub-model"

        sel = _run(
            dispatcher.select_team(
                user_message="annotate with scGPT",
                available_templates=AVAILABLE,
                default_template_id="default",
                preferred_model="stub-model",
            )
        )
        assert isinstance(sel, TeamSelection)
        assert sel.template_id == "single_cell_team"
        assert sel.confidence == 0.9

    def test_unknown_id_falls_back_to_default(self):
        dispatcher = TeamDispatcher()
        dispatcher._dispatch_agent = _mock_agent(
            json.dumps({"template_id": "not_a_real_team", "reason": "?", "confidence": 0.5})
        )
        dispatcher._dispatch_agent_model = "stub-model"

        sel = _run(
            dispatcher.select_team(
                "hello",
                AVAILABLE,
                "default",
                preferred_model="stub-model",
            )
        )
        assert sel.template_id == "default"
        assert sel.confidence == 0.0
        assert "not_a_real_team" in sel.reason

    def test_malformed_json_falls_back_to_default(self):
        dispatcher = TeamDispatcher()
        dispatcher._dispatch_agent = _mock_agent("this is not json at all")
        dispatcher._dispatch_agent_model = "stub-model"

        sel = _run(
            dispatcher.select_team(
                "hello",
                AVAILABLE,
                "default",
                preferred_model="stub-model",
            )
        )
        assert sel.template_id == "default"
        assert sel.confidence == 0.0

    def test_llm_error_falls_back_silently(self):
        dispatcher = TeamDispatcher()
        dispatcher._dispatch_agent = _mock_agent(RuntimeError("boom"))
        dispatcher._dispatch_agent_model = "stub-model"

        sel = _run(
            dispatcher.select_team(
                "hello",
                AVAILABLE,
                "default",
                preferred_model="stub-model",
            )
        )
        assert sel.template_id == "default"
        assert sel.confidence == 0.0
        assert "dispatcher error" in sel.reason

    def test_empty_message_short_circuits(self):
        dispatcher = TeamDispatcher()
        dispatcher._dispatch_agent = _mock_agent("should not be called")
        dispatcher._dispatch_agent_model = "stub-model"

        sel = _run(
            dispatcher.select_team(
                "   ",
                AVAILABLE,
                "default",
                preferred_model="stub-model",
            )
        )
        assert sel.template_id == "default"
        assert sel.confidence == 0.0
        dispatcher._dispatch_agent.run.assert_not_called()

    def test_no_templates_short_circuits(self):
        dispatcher = TeamDispatcher()
        dispatcher._dispatch_agent = _mock_agent("should not be called")
        dispatcher._dispatch_agent_model = "stub-model"

        sel = _run(
            dispatcher.select_team(
                "hello",
                [],
                "default",
                preferred_model="stub-model",
            )
        )
        assert sel.template_id == "default"
        dispatcher._dispatch_agent.run.assert_not_called()

    def test_confidence_out_of_range_is_clamped(self):
        dispatcher = TeamDispatcher()
        dispatcher._dispatch_agent = _mock_agent(
            json.dumps({"template_id": "omicverse_team", "reason": "x", "confidence": 42})
        )
        dispatcher._dispatch_agent_model = "stub-model"

        sel = _run(
            dispatcher.select_team(
                "omics analysis",
                AVAILABLE,
                "default",
                preferred_model="stub-model",
            )
        )
        assert sel.template_id == "omicverse_team"
        assert 0.0 <= sel.confidence <= 1.0

    def test_exactly_one_llm_call_per_invocation(self):
        dispatcher = TeamDispatcher()
        mock = _mock_agent(
            json.dumps({"template_id": "single_cell_team", "reason": "ok", "confidence": 0.8})
        )
        dispatcher._dispatch_agent = mock
        dispatcher._dispatch_agent_model = "stub-model"

        _run(
            dispatcher.select_team(
                "scGPT request",
                AVAILABLE,
                "default",
                preferred_model="stub-model",
            )
        )
        assert mock.run.call_count == 1
