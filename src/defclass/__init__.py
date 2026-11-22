"""defclass: SOTA zero-shot generative text classification with self-sufficient definitions."""

from .classifier import ZeroShotClassifier
from .definitions import audit_and_repair_one, generate_definitions
from .labels import detect_opaque_labels, is_opaque_label
from .providers import (
    AnthropicProvider,
    DecisionProvider,
    GeminiDecisionProvider,
    GeminiProvider,
    JevDecisionProvider,
    LLMProvider,
    LiteLLMProvider,
    OpenAICompatibleProvider,
    create_provider,
)
from .types import (
    ClassDefinition,
    DecisionResult,
    OptimizationReport,
    OptimizationStep,
    Prediction,
)

__all__ = [
    "AnthropicProvider",
    "ClassDefinition",
    "DecisionProvider",
    "DecisionResult",
    "GeminiDecisionProvider",
    "GeminiProvider",
    "JevDecisionProvider",
    "LLMProvider",
    "LiteLLMProvider",
    "OpenAICompatibleProvider",
    "OptimizationReport",
    "OptimizationStep",
    "Prediction",
    "ZeroShotClassifier",
    "audit_and_repair_one",
    "create_provider",
    "detect_opaque_labels",
    "generate_definitions",
    "is_opaque_label",
]
