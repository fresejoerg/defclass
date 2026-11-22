"""Core data models and schemas for defclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class ClassDefinitionSchema(BaseModel):
    """Structured, self-sufficient definition of a single class."""

    semantic_name: str = Field(
        description="Concise, human-readable title for this class."
    )
    definition: str = Field(
        description=(
            "Self-contained 2-3 sentence definition understandable in isolation "
            "without seeing other classes."
        )
    )
    inclusion_criteria: list[str] = Field(
        description="3-5 concrete conditions or signals indicating membership."
    )
    exclusion_criteria: list[str] = Field(
        description="3-5 boundary conditions ruling out adjacent or confusable intents."
    )
    representative_patterns: list[str] = Field(
        description="3-5 generalized phrasing patterns illustrating this class."
    )


class DefinitionAuditSchema(BaseModel):
    """Isolation audit result for a single class definition."""

    is_self_sufficient: bool
    has_external_label_refs: bool
    issues: list[str] = Field(default_factory=list)
    revised_definition: ClassDefinitionSchema | None = None


class ClusterSummarySchema(BaseModel):
    """Self-sufficient summary definition for a cluster of classes."""

    summary_name: str
    summary_definition: str
    key_signals: list[str]


@dataclass(frozen=True)
class ClassDefinition:
    """Normalized class definition stored and used by ZeroShotClassifier."""

    label: str
    semantic_name: str
    definition: str
    inclusion_criteria: list[str]
    exclusion_criteria: list[str]
    representative_patterns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "semantic_name": self.semantic_name,
            "definition": self.definition,
            "inclusion_criteria": list(self.inclusion_criteria),
            "exclusion_criteria": list(self.exclusion_criteria),
            "representative_patterns": list(self.representative_patterns),
        }

    @classmethod
    def from_dict(cls, label: str, data: dict[str, Any]) -> ClassDefinition:
        return cls(
            label=label,
            semantic_name=str(data.get("semantic_name", label)),
            definition=str(data["definition"]),
            inclusion_criteria=list(data.get("inclusion_criteria", [])),
            exclusion_criteria=list(data.get("exclusion_criteria", [])),
            representative_patterns=list(data.get("representative_patterns", [])),
        )

    def render(self, display_label: str, include_semantic_name: bool = False) -> str:
        """Render the definition block for prompting."""
        header = (
            f"Class: {display_label} ({self.semantic_name})"
            if include_semantic_name
            else f"Class: {display_label}"
        )
        sections = [header, f"Definition: {self.definition}"]
        if self.inclusion_criteria:
            inc = "\n".join(f"  - {item}" for item in self.inclusion_criteria)
            sections.append(f"Belongs to this class if:\n{inc}")
        if self.exclusion_criteria:
            exc = "\n".join(f"  - {item}" for item in self.exclusion_criteria)
            sections.append(f"Does NOT belong to this class if:\n{exc}")
        if self.representative_patterns:
            pat = "\n".join(f"  - {item}" for item in self.representative_patterns)
            sections.append(f"Typical phrasing patterns:\n{pat}")
        return "\n".join(sections)


@dataclass(frozen=True)
class DecisionResult:
    """Normalized output from a decision-model provider."""

    choice: str
    confidence: float | None = None
    probabilities: dict[str, float] | None = None


@dataclass(frozen=True)
class Prediction:
    """Classification output returned by ZeroShotClassifier."""

    label: str
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    rationale: str | None = None
    votes: list[str] = field(default_factory=list)
    routed_clusters: list[str] | None = None
    latency_ms: float = 0.0
    prompt_chars: int = 0


@dataclass(frozen=True)
class OptimizationStep:
    """Single iteration record from reflective definition optimization."""

    iteration: int
    dev_accuracy: float
    dev_macro_f1: float
    mutated_labels: list[str]
    accepted: bool


@dataclass(frozen=True)
class OptimizationReport:
    """Summary of a reflective definition optimization run."""

    initial_accuracy: float
    initial_macro_f1: float
    best_accuracy: float
    best_macro_f1: float
    history: list[OptimizationStep]
