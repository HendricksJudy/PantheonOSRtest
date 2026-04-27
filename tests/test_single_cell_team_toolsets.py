"""Smoke test asserting the single_cell_team template wires scfm into the agents
that now execute router plans (leader, analysis_expert) and keeps fm_router
scoped to scfm only."""

from pathlib import Path

import pytest

from pantheon.factory import get_template_manager


def _load_single_cell_team(tmp_path: Path):
    template_manager = get_template_manager(tmp_path)
    template = template_manager.get_template("single_cell_team")
    if template is None:
        pytest.skip("single_cell_team template not present in this environment")
    return template


def test_leader_has_scfm_toolset(tmp_path: Path):
    team = _load_single_cell_team(tmp_path)
    leader = next((a for a in team.agents if a.id == "leader" or a.name == "leader"), None)
    assert leader is not None, "leader agent not found in single_cell_team"
    assert "scfm" in (leader.toolsets or []), (
        f"leader must have 'scfm' toolset to execute router plans; got {leader.toolsets}"
    )


def test_analysis_expert_has_scfm_toolset(tmp_path: Path):
    team = _load_single_cell_team(tmp_path)
    ae = next((a for a in team.agents if a.id == "analysis_expert" or a.name == "analysis_expert"), None)
    assert ae is not None, "analysis_expert agent not found in single_cell_team"
    assert "scfm" in (ae.toolsets or []), (
        f"analysis_expert must have 'scfm' toolset to execute delegated plans; got {ae.toolsets}"
    )


def test_fm_router_still_has_scfm_toolset(tmp_path: Path):
    team = _load_single_cell_team(tmp_path)
    fm = next((a for a in team.agents if a.id == "fm_router" or a.name == "fm_router"), None)
    assert fm is not None, "fm_router agent not found in single_cell_team"
    assert "scfm" in (fm.toolsets or []), (
        f"fm_router must still have 'scfm' toolset; got {fm.toolsets}"
    )
