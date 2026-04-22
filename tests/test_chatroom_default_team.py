"""Tests for default_team_template setting and dispatcher integration in ChatRoom."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from pantheon.chatroom.room import ChatRoom
from pantheon.chatroom.special_agents import TeamSelection
from pantheon.factory import get_template_manager
from pantheon.internal.memory import MemoryManager


def _make_chatroom(tmp_path: Path) -> ChatRoom:
    chatroom = ChatRoom.__new__(ChatRoom)
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    chatroom.memory_manager = MemoryManager(memory_dir, use_jsonl=True)
    chatroom.template_manager = get_template_manager(tmp_path)
    chatroom.chat_teams = {}
    chatroom._default_team = None
    return chatroom


def _make_memory():
    """Build a minimal object that quacks like an internal.memory.Memory instance
    for the fields _maybe_dispatch_team_for_new_chat touches."""

    class FakeMemory:
        def __init__(self):
            self.extra_data = {}

        def set_metadata(self, key, value):
            self.extra_data[key] = value

        def set_metadata_in_memory(self, key, value):
            self.extra_data[key] = value

    return FakeMemory()


class TestMaybeDispatchTeamForNewChat:
    @pytest.mark.asyncio
    async def test_no_op_when_team_already_stored(self, tmp_path: Path):
        chatroom = _make_chatroom(tmp_path)
        memory = _make_memory()
        memory.extra_data["team_template"] = {"id": "pre_existing"}
        notice = await chatroom._maybe_dispatch_team_for_new_chat(memory, "annotate scGPT")
        assert notice is None
        # Memory remains untouched
        assert memory.extra_data["team_template"] == {"id": "pre_existing"}

    @pytest.mark.asyncio
    async def test_no_op_when_message_empty(self, tmp_path: Path):
        chatroom = _make_chatroom(tmp_path)
        memory = _make_memory()
        notice = await chatroom._maybe_dispatch_team_for_new_chat(memory, "   ")
        assert notice is None
        assert "team_template" not in memory.extra_data

    @pytest.mark.asyncio
    async def test_no_op_when_setting_disabled(self, tmp_path: Path, monkeypatch):
        chatroom = _make_chatroom(tmp_path)
        memory = _make_memory()
        # Disable via settings patch
        from pantheon import settings as settings_mod
        settings = settings_mod.get_settings()
        monkeypatch.setattr(type(settings), "team_dispatcher_enabled", property(lambda self: False))
        notice = await chatroom._maybe_dispatch_team_for_new_chat(memory, "annotate scGPT")
        assert notice is None
        assert "team_template" not in memory.extra_data

    @pytest.mark.asyncio
    async def test_no_op_when_dispatcher_picks_default(self, tmp_path: Path):
        chatroom = _make_chatroom(tmp_path)
        memory = _make_memory()

        with patch(
            "pantheon.chatroom.special_agents.TeamDispatcher.select_team",
            new=AsyncMock(return_value=TeamSelection("default", "generic", 0.2)),
        ):
            notice = await chatroom._maybe_dispatch_team_for_new_chat(memory, "hello")
        assert notice is None
        assert "team_template" not in memory.extra_data

    @pytest.mark.asyncio
    async def test_persists_selected_template_and_returns_notice(self, tmp_path: Path):
        chatroom = _make_chatroom(tmp_path)
        memory = _make_memory()

        # Ensure the single_cell_team template is discoverable. If unavailable
        # (e.g. factory template missing), skip this test instead of failing.
        templates = chatroom.template_manager.list_templates()
        ids = {getattr(t, "id", None) for t in templates}
        if "single_cell_team" not in ids:
            pytest.skip("single_cell_team template not present in this environment")

        with patch(
            "pantheon.chatroom.special_agents.TeamDispatcher.select_team",
            new=AsyncMock(
                return_value=TeamSelection(
                    "single_cell_team", "mentions scGPT", 0.9
                )
            ),
        ):
            notice = await chatroom._maybe_dispatch_team_for_new_chat(
                memory, "annotate my .h5ad with scGPT"
            )

        assert notice is not None
        assert notice["template_id"] == "single_cell_team"
        assert "scGPT" in notice["reason"]
        assert memory.extra_data["team_template"]["id"] == "single_cell_team"
