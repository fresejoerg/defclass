"""Automatic self-sufficient class definition synthesis and isolation auditing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
from typing import Any

from .labels import is_opaque_label
from .providers import LLMProvider
from .types import (
    ClassDefinition,
    ClassDefinitionSchema,
    DefinitionAuditSchema,
)


def _synthesize_one(
    provider: LLMProvider,
    label: str,
    examples: list[str],
    all_labels: list[str],
    domain_context: str | None,
) -> ClassDefinition:
    """Synthesize a structured, self-sufficient definition for one class."""
    domain_line = f"Domain context: {domain_context}\n" if domain_context else ""
    if examples:
        ex_block = "\n".join(f"- {ex}" for ex in examples)
    else:
        ex_block = "(No example utterances provided; infer solely from the label name and domain context.)"

    named_siblings = [lbl for lbl in all_labels if not is_opaque_label(lbl) and lbl != label]
    sibling_note = ""
    if named_siblings and len(named_siblings) <= 80:
        sibling_note = (
            f"Other classes in this taxonomy (for boundary awareness only; DO NOT mention them): "
            f"{', '.join(named_siblings)}\n"
        )

    prompt = (
        f"You are designing a zero-shot text classification specification.\n"
        f"{domain_line}"
        f"Target class label: {label!r}\n"
        f"{sibling_note}\n"
        f"Representative examples belonging to {label!r}:\n{ex_block}\n\n"
        f"Write a comprehensive, SELF-SUFFICIENT class definition for {label!r}. "
        f"An LLM or human reading ONLY this single definition in isolation must be able to "
        f"accurately decide whether any utterance belongs to this class.\n"
        f"- Ground the definition in the user's underlying intent or semantic category.\n"
        f"- In inclusion_criteria, list concrete signals and conditions for membership.\n"
        f"- In exclusion_criteria, rule out adjacent or easily confused intents in plain natural language.\n"
        f"- NEVER mention other class labels, dataset names, or 'training examples'."
    )
    raw = provider.complete_json(
        prompt,
        ClassDefinitionSchema.model_json_schema(),
        temperature=0.2,
    )
    parsed = ClassDefinitionSchema.model_validate(raw)
    return ClassDefinition(
        label=label,
        semantic_name=parsed.semantic_name,
        definition=parsed.definition,
        inclusion_criteria=parsed.inclusion_criteria,
        exclusion_criteria=parsed.exclusion_criteria,
        representative_patterns=parsed.representative_patterns,
    )


def audit_and_repair_one(
    provider: LLMProvider,
    definition: ClassDefinition,
) -> ClassDefinition:
    """Audit a definition in isolation for self-sufficiency and label leaks; repair if needed."""
    def_json = json.dumps(
        {
            "semantic_name": definition.semantic_name,
            "definition": definition.definition,
            "inclusion_criteria": definition.inclusion_criteria,
            "exclusion_criteria": definition.exclusion_criteria,
            "representative_patterns": definition.representative_patterns,
        },
        indent=2,
        ensure_ascii=False,
    )
    prompt = (
        f"Audit this class definition against two strict requirements:\n"
        f"1. SELF-SUFFICIENCY: An LLM or human reading ONLY this definition in isolation can "
        f"accurately decide whether an utterance belongs to this class.\n"
        f"2. NO EXTERNAL REFERENCES: The text must NOT reference other class names, internal "
        f"label IDs (e.g., snake_case/kebab-case/numeric codes), dataset names, or phrases like "
        f"'other classes' or 'training examples'. Plain-language exclusions are good and expected.\n\n"
        f"Definition:\n{def_json}\n\n"
        f"If both checks pass, set is_self_sufficient=true, has_external_label_refs=false, "
        f"issues=[], and revised_definition=null. Otherwise provide a corrected revised_definition."
    )
    raw = provider.complete_json(
        prompt,
        DefinitionAuditSchema.model_json_schema(),
        temperature=0.0,
    )
    audit = DefinitionAuditSchema.model_validate(raw)
    if (
        (not audit.is_self_sufficient or audit.has_external_label_refs)
        and audit.revised_definition is not None
    ):
        rev = audit.revised_definition
        return ClassDefinition(
            label=definition.label,
            semantic_name=rev.semantic_name,
            definition=rev.definition,
            inclusion_criteria=rev.inclusion_criteria,
            exclusion_criteria=rev.exclusion_criteria,
            representative_patterns=rev.representative_patterns,
        )
    return definition


def generate_definitions(
    provider: LLMProvider,
    labels: list[str],
    *,
    examples: dict[str, list[str]] | None = None,
    domain_context: str | None = None,
    examples_per_class: int = 10,
    audit: bool = True,
    auditor_provider: LLMProvider | None = None,
    max_workers: int = 8,
    seed: int = 42,
) -> dict[str, ClassDefinition]:
    """Synthesize (and optionally isolation-audit) self-sufficient definitions for all labels."""
    rng = random.Random(seed)
    sampled: dict[str, list[str]] = {}
    for lbl in labels:
        exs = (examples or {}).get(lbl, [])
        sampled[lbl] = rng.sample(exs, examples_per_class) if len(exs) > examples_per_class else list(exs)

    defs: dict[str, ClassDefinition] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(labels)))) as pool:
        futs = {
            pool.submit(_synthesize_one, provider, lbl, sampled[lbl], labels, domain_context): lbl
            for lbl in labels
        }
        for fut in as_completed(futs):
            lbl = futs[fut]
            defs[lbl] = fut.result()

    if audit:
        auditor = auditor_provider or provider
        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(labels)))) as pool:
            futs = {
                pool.submit(audit_and_repair_one, auditor, defs[lbl]): lbl
                for lbl in labels
            }
            for fut in as_completed(futs):
                lbl = futs[fut]
                defs[lbl] = fut.result()

    return {lbl: defs[lbl] for lbl in labels}
