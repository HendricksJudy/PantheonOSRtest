# GUI Router Regression Checklist (One-Click + Manual)

This checklist verifies the GUI-to-SCFM routing path:

1. Create a fresh chat with no pre-bound template.
2. On first user message, auto-dispatch to `single_cell_team`.
3. Validate and normalize `fm_router` plan output using `scfm_validate_plan`.

## One-click deterministic smoke test

```bash
python scripts/gui_router_regression.py --verbose
```

Expected output includes:

- `dispatch.selected_template == "single_cell_team"`
- `validate_plan.normalized_tool == "scfm_run"`
- `all_passed == true`

## Optional manual GUI checks

1. Open GUI and create a new chat (without explicitly selecting template).
2. First message: ask for scFM task such as scGPT embedding/annotation.
3. Confirm the chat template switches to single-cell team.
4. Confirm plan is validated before execution (`scfm_validate_plan`).
5. If `questions` are present in plan, ensure they are surfaced before execution.
