#!/usr/bin/env python3
"""One-click regression checks for GUI -> team dispatch -> fm_router plan validation flow.

This script is intentionally lightweight and deterministic so it can run in CI/dev boxes
without live LLM credentials or browser automation.

Checks:
1) Simulate first-message auto-dispatch and assert chat template becomes single_cell_team.
2) Validate/normalize an fm_router-like plan where a model name is mistakenly used as tool.

Exit code 0 on success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pantheon.chatroom.room import ChatRoom
from pantheon.chatroom.special_agents import TeamSelection
from pantheon.factory import get_template_manager
from pantheon.internal.memory import MemoryManager
from pantheon.toolsets.scfm.toolset import SCFMToolSet


async def _build_ephemeral_chatroom(tmp_root: Path) -> ChatRoom:
    chatroom = ChatRoom.__new__(ChatRoom)
    memory_dir = tmp_root / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    chatroom.memory_manager = MemoryManager(memory_dir, use_jsonl=True)
    chatroom.template_manager = get_template_manager(tmp_root)
    chatroom.chat_teams = {}
    chatroom._default_team = None
    return chatroom


async def check_auto_dispatch_to_single_cell_team(verbose: bool = False) -> dict:
    with TemporaryDirectory() as td:
        tmp_root = Path(td)
        chatroom = await _build_ephemeral_chatroom(tmp_root)

        created = await chatroom.create_chat(chat_name="gui-regression")
        if not created.get("success"):
            return {"ok": False, "error": f"create_chat failed: {created}"}

        chat_id = created["chat_id"]
        memory = await asyncio.to_thread(chatroom.memory_manager.get_memory, chat_id)

        if memory.extra_data.get("team_template") is not None:
            return {
                "ok": False,
                "error": "expected no pre-bound team_template for a fresh chat",
            }

        with patch(
            "pantheon.chatroom.special_agents.TeamDispatcher.select_team",
            new=AsyncMock(
                return_value=TeamSelection(
                    template_id="single_cell_team",
                    reason="message indicates scFM task",
                    confidence=0.97,
                )
            ),
        ):
            notice = await chatroom._maybe_dispatch_team_for_new_chat(
                memory,
                "Please run scGPT embedding on my h5ad dataset",
            )

        selected_id = memory.extra_data.get("team_template", {}).get("id")
        ok = selected_id == "single_cell_team" and notice and notice.get("template_id") == "single_cell_team"
        result = {
            "ok": bool(ok),
            "selected_template": selected_id,
            "notice": notice,
        }
        if not ok:
            result["error"] = "dispatcher did not persist single_cell_team"
        if verbose:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        return result


async def check_router_plan_validation(verbose: bool = False) -> dict:
    toolset = SCFMToolSet()

    router_like_plan = {
        "intent": {"task": "embed", "confidence": 0.9, "constraints": {}},
        "inputs": {"query": "Embed with scgpt", "adata_path": "/tmp/demo.h5ad"},
        "data_profile": None,
        "selection": {
            "recommended": {"name": "scgpt", "rationale": "general embedding"},
            "fallbacks": [],
        },
        "resolved_params": {},
        "plan": [
            {
                "tool": "scgpt",
                "args": {"task": "embed", "adata_path": "/tmp/demo.h5ad"},
            }
        ],
        "questions": [],
        "warnings": [],
    }

    validated = await toolset.scfm_validate_plan(router_like_plan)
    normalized_step = (validated.get("normalized_plan") or {}).get("plan", [{}])[0]

    ok = (
        validated.get("ok") is True
        and normalized_step.get("tool") == "scfm_run"
        and normalized_step.get("args", {}).get("model_name") == "scgpt"
    )

    result = {
        "ok": bool(ok),
        "validate_ok": validated.get("ok"),
        "normalized_tool": normalized_step.get("tool"),
        "normalized_model_name": normalized_step.get("args", {}).get("model_name"),
        "errors": validated.get("errors", []),
    }
    if not ok:
        result["error"] = "scfm_validate_plan did not normalize model-name tool to scfm_run"
    if verbose:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


async def run(verbose: bool = False) -> int:
    dispatch_result = await check_auto_dispatch_to_single_cell_team(verbose=verbose)
    validate_result = await check_router_plan_validation(verbose=verbose)

    summary = {
        "dispatch": dispatch_result,
        "validate_plan": validate_result,
        "all_passed": dispatch_result.get("ok") and validate_result.get("ok"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["all_passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GUI->router regression smoke checks")
    parser.add_argument("--verbose", action="store_true", help="Print per-check details")
    args = parser.parse_args()
    return asyncio.run(run(verbose=args.verbose))


if __name__ == "__main__":
    sys.exit(main())
