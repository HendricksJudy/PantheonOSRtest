"""
SCFM Router utilities — Pydantic models, normalization, validation, and model-card
formatting shared by the fm_router sub-agent template and the scfm_validate_plan tool.

The deprecated `scfm_router` tool has been removed; these helpers remain as a pure
library so plans produced by the fm_router sub-agent can be validated and normalized
before the leader/analysis_expert agent executes them.
"""

import json
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from .registry import (
    GeneIDScheme,
    ModelSpec,
    SkillReadyStatus,
    TaskType,
    get_registry,
)


# =============================================================================
# Constants
# =============================================================================

VALID_TASKS = [t.value for t in TaskType]

VALID_SCFM_TOOLS = [
    "scfm_profile_data",
    "scfm_preprocess_validate",
    "scfm_run",
    "scfm_interpret_results",
    "scfm_list_models",
    "scfm_describe_model",
    "scfm_select_model",
]


def _normalize_router_output_dict(output_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize common LLM plan mistakes into valid SCFM router output.

    In practice the router sometimes emits a model name such as ``scplantllm``
    as the plan tool. Convert those steps into ``scfm_run`` with
    ``model_name=<that model>`` while preserving the original args.
    """
    if not isinstance(output_dict, dict):
        return output_dict

    normalized = json.loads(json.dumps(output_dict))
    registry = get_registry()
    plan = normalized.get("plan")
    if not isinstance(plan, list):
        return normalized

    for step in plan:
        if not isinstance(step, dict):
            continue
        tool_name = step.get("tool")
        if not isinstance(tool_name, str) or tool_name in VALID_SCFM_TOOLS:
            continue
        if registry.get(tool_name.lower()) is None:
            continue

        args = step.get("args")
        if not isinstance(args, dict):
            args = {}
            step["args"] = args
        args.setdefault("model_name", tool_name.lower())
        step["tool"] = "scfm_run"

    return normalized

ROUTER_SYSTEM_PROMPT = """You are an expert scFM (single-cell foundation model) router.
Your job is to analyze user queries about single-cell data analysis and determine:
1. Which task they want to perform (embed, integrate, annotate, spatial, perturb, drug_response)
2. Which model best fits their needs from the available registry
3. What parameters are needed for execution
4. If any clarification is required

IMPORTANT: You MUST respond with valid JSON only. No markdown, no explanation text outside JSON.

## Available Tasks
- embed: Generate cell embeddings using a foundation model
- integrate: Batch integration / correction using foundation model embeddings
- annotate: Cell type annotation (may require fine-tuning depending on model)
- spatial: Spatial transcriptomics analysis (requires spatial coordinates)
- perturb: Perturbation prediction / analysis
- drug_response: Drug response prediction

## Output Format
Return a JSON object with this exact structure:
{
  "intent": {
    "task": "<task_name>",
    "confidence": <0.0-1.0>,
    "constraints": {}
  },
  "inputs": {
    "query": "<original_query>",
    "adata_path": "<path_if_provided>"
  },
  "data_profile": <null_or_profile_object>,
  "selection": {
    "recommended": {"name": "<model_name>", "rationale": "<why>"},
    "fallbacks": [{"name": "<model_name>", "rationale": "<why>"}]
  },
  "resolved_params": {
    "output_path": "<path_or_null>",
    "batch_key": "<key_or_null>",
    "label_key": "<key_or_null>"
  },
  "plan": [
    {"tool": "<tool_name>", "args": {}}
  ],
  "questions": [
    {"field": "<param_name>", "question": "<clarification_question>", "options": []}
  ],
  "warnings": []
}

## CRITICAL: Model Selection Rules

**Match the user's specific requirements to each model's unique differentiator and "Use when" guidance.**
Do NOT default to any single model. Each model has a distinct strength — select based on what the user actually needs.

### Disambiguation Table (confusable models)
| User mentions...                          | Select         | NOT              |
|-------------------------------------------|---------------|------------------|
| multi-omics, CITE-seq, RNA+ATAC+Protein   | scmulan       | scgpt            |
| spatial transcriptomics, niche, Visium    | nicheformer   | scgpt            |
| ATAC-seq only, chromatin accessibility    | atacformer    | scgpt/scmulan    |
| denoising, ambient RNA, protein-coding    | scprint       | scgpt            |
| unsupervised clustering, label-free       | aidocell      | scgpt            |
| cell-cell communication, multicellular    | pulsar        | scgpt            |
| fast inference, high throughput, million+  | cellplm       | scgpt            |
| next-token, autoregressive, generative    | tgpt          | scgpt/geneformer |
| MLP architecture, largest scale           | cellfm        | scgpt            |
| compact 200-dim, lightweight              | scbert        | scgpt            |
| ontology, hierarchical cell types         | sccello       | scgpt            |
| plant, polyploidy, Arabidopsis            | scplantllm    | scgpt/uce        |
| text+cell alignment, NL cell queries      | langcell      | scgpt            |
| LLM fine-tuning, cells-as-text            | cell2sentence | scgpt            |
| gene-level (not cell), no GPU, API-based  | genept        | scgpt/geneformer |
| chat-based, conversational annotation     | chatcell      | scgpt            |
| Ensembl IDs, network biology, CPU-only    | geneformer    | scgpt            |
| cross-species, zebrafish/frog/pig/macaque | uce           | scgpt/geneformer |
| prior knowledge, gene regulatory networks | genecompass   | scgpt            |
| perturbation prediction (gene KO/KD)      | tabula        | scfoundation/scgpt |
| general RNA embed/integrate, no special needs | scgpt     | -                |

### Selection Priority
1. **Unique requirement match**: If the query mentions a specific capability listed in a model's "Use when" field, select that model — even if it is ⚠️ partial-spec.
2. **Modality/species match**: ATAC-only → atacformer. Plant → scplantllm. Multi-omics → scmulan. Non-standard species → uce.
3. **Task-specific match**: Zero-shot annotation → sccello or chatcell. Perturbation → tabula. Spatial → nicheformer.
4. **General fallback**: Only select scgpt or geneformer when no specific differentiating requirement is present.

### Rules
1. Always select models from the provided model cards
2. If uncertain about parameters (like batch_key), add a question
3. If data profile shows incompatibility, select alternative model or add warning
4. Generate a complete execution plan with tool calls
5. Set confidence based on how clear the user's intent is
6. Skill-ready status (✅ vs ⚠️) is about adapter documentation, NOT model quality — do not prefer ✅ models over ⚠️ models based on status alone
"""


# =============================================================================
# Data Models (Pydantic)
# =============================================================================


class RouterIntent(BaseModel):
    """Inferred task intent from user query."""
    task: str = Field(..., description="The inferred task type")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in task inference")
    constraints: dict[str, Any] = Field(default_factory=dict, description="Additional constraints from query")

    @field_validator("task")
    @classmethod
    def validate_task(cls, v: str) -> str:
        if v not in VALID_TASKS:
            raise ValueError(f"Invalid task: {v}. Must be one of {VALID_TASKS}")
        return v


class RouterInputs(BaseModel):
    """Input information from user query."""
    query: str = Field(..., description="Original user query")
    adata_path: Optional[str] = Field(default=None, description="Path to AnnData file if provided")


class ModelSelection(BaseModel):
    """Model selection with rationale."""
    name: str = Field(..., description="Model name")
    rationale: str = Field(default="", description="Why this model was selected")


class RouterSelection(BaseModel):
    """Model selection output."""
    recommended: ModelSelection = Field(..., description="Recommended model")
    fallbacks: list[ModelSelection] = Field(default_factory=list, description="Fallback options")


class ResolvedParams(BaseModel):
    """Resolved parameters for execution."""
    output_path: Optional[str] = Field(default=None, description="Output file path")
    batch_key: Optional[str] = Field(default=None, description="Batch key in .obs")
    label_key: Optional[str] = Field(default=None, description="Label key in .obs")


class ToolCall(BaseModel):
    """A single tool call in the execution plan."""
    tool: str = Field(..., description="Tool name to call")
    args: dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")

    @field_validator("tool")
    @classmethod
    def validate_tool(cls, v: str) -> str:
        if v not in VALID_SCFM_TOOLS:
            raise ValueError(f"Invalid tool: {v}. Must be one of {VALID_SCFM_TOOLS}")
        return v


class Question(BaseModel):
    """A clarifying question for the user."""
    field: str = Field(..., description="Parameter field this question is about")
    question: str = Field(..., description="The question to ask")
    options: list[str] = Field(default_factory=list, description="Suggested options if applicable")


class RouterOutput(BaseModel):
    """Complete router output."""
    intent: RouterIntent = Field(..., description="Inferred task intent")
    inputs: RouterInputs = Field(..., description="Input information")
    data_profile: Optional[dict[str, Any]] = Field(default=None, description="Data profile if adata was provided")
    selection: RouterSelection = Field(..., description="Model selection")
    resolved_params: ResolvedParams = Field(default_factory=ResolvedParams, description="Resolved parameters")
    plan: list[ToolCall] = Field(default_factory=list, description="Execution plan")
    questions: list[Question] = Field(default_factory=list, description="Clarifying questions")
    warnings: list[str] = Field(default_factory=list, description="Warnings about the request")


# =============================================================================
# Validation
# =============================================================================


def validate_router_output(output_dict: dict[str, Any]) -> tuple[bool, list[str], Optional[RouterOutput]]:
    """
    Validate router output against schema and registry.

    Args:
        output_dict: Parsed JSON output from LLM

    Returns:
        Tuple of (is_valid, error_messages, parsed_output)
    """
    errors = []
    output_dict = _normalize_router_output_dict(output_dict)
    registry = get_registry()

    # Validate against Pydantic schema
    try:
        parsed = RouterOutput.model_validate(output_dict)
    except Exception as e:
        errors.append(f"Schema validation error: {str(e)}")
        return False, errors, None

    # Validate model exists in registry
    recommended_model = parsed.selection.recommended.name.lower()
    if registry.get(recommended_model) is None:
        available = [m.name for m in registry.list_models()]
        errors.append(f"Model '{recommended_model}' not found in registry. Available: {available[:10]}")

    # Validate fallback models exist
    for fallback in parsed.selection.fallbacks:
        if registry.get(fallback.name.lower()) is None:
            errors.append(f"Fallback model '{fallback.name}' not found in registry")

    # Validate tool names in plan
    for tool_call in parsed.plan:
        if tool_call.tool not in VALID_SCFM_TOOLS:
            errors.append(f"Invalid tool '{tool_call.tool}' in plan. Valid: {VALID_SCFM_TOOLS}")

    if errors:
        return False, errors, parsed

    return True, [], parsed


# =============================================================================
# Prompt Builder
# =============================================================================


def build_model_cards(
    skill_ready_only: bool = False,
    max_vram_gb: Optional[int] = None,
    prefer_zero_shot: bool = True,
) -> str:
    """
    Build formatted model cards for LLM prompt.

    Args:
        skill_ready_only: Only include skill-ready models
        max_vram_gb: Filter by max VRAM constraint
        prefer_zero_shot: Highlight zero-shot capable models

    Returns:
        Formatted string of model cards
    """
    registry = get_registry()
    models = registry.list_models(skill_ready_only=skill_ready_only)

    if max_vram_gb:
        models = [m for m in models if m.hardware.min_vram_gb <= max_vram_gb]

    cards = []
    for spec in models:
        status_icon = "✅" if spec.skill_ready == SkillReadyStatus.READY else "⚠️"
        zero_shot_note = " [zero-shot]" if spec.zero_shot_embedding else ""

        card = f"""### {status_icon} {spec.name}{zero_shot_note}
- **Version**: {spec.version}
- **Tasks**: {', '.join(t.value for t in spec.tasks)}
- **Species**: {', '.join(spec.species)}
- **Gene IDs**: {spec.gene_id_scheme.value}
- **VRAM**: {spec.hardware.min_vram_gb}GB min
- **CPU fallback**: {"Yes" if spec.hardware.cpu_fallback else "No"}
- **Differentiator**: {spec.differentiator or "General-purpose"}
- **Use when**: {spec.prefer_when or "No specific preference"}
"""
        cards.append(card)

    return "\n".join(cards)
