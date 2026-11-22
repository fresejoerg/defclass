"""Reflective prompt/definition optimization (built-in hill-climbing + optional DSPy wrapper)."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import warnings
from typing import Any, Callable

from .providers import LLMProvider
from .types import (
    ClassDefinition,
    ClassDefinitionSchema,
    OptimizationReport,
    OptimizationStep,
)


def _accuracy_and_macro_f1(y_true: list[str], y_pred: list[str]) -> tuple[float, float]:
    """Compute Accuracy and Macro-F1 without requiring scikit-learn."""
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n

    labels = sorted(set(y_true))
    f1s: list[float] = []
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    macro_f1 = sum(f1s) / max(len(f1s), 1)
    return acc, macro_f1


def _reflect_and_refine(
    optimizer_provider: LLMProvider,
    target_def: ClassDefinition,
    all_defs: dict[str, ClassDefinition],
    fn_cases: list[tuple[str, str]],
    fp_cases: list[tuple[str, str]],
) -> ClassDefinition:
    """Use natural-language reflection on dev errors to sharpen a ClassDefinition."""
    cur_json = json.dumps(
        {
            "semantic_name": target_def.semantic_name,
            "definition": target_def.definition,
            "inclusion_criteria": target_def.inclusion_criteria,
            "exclusion_criteria": target_def.exclusion_criteria,
            "representative_patterns": target_def.representative_patterns,
        },
        indent=2,
        ensure_ascii=False,
    )
    fn_block = "\n".join(
        f'- "{text}" (mistakenly predicted as {all_defs[pred].semantic_name!r})'
        for text, pred in fn_cases[:6]
        if pred in all_defs
    ) or "(none)"
    fp_block = "\n".join(
        f'- "{text}" (actually belongs to {all_defs[truth].semantic_name!r})'
        for text, truth in fp_cases[:6]
        if truth in all_defs
    ) or "(none)"

    prompt = (
        f"You are optimizing a zero-shot class definition based on classification errors.\n\n"
        f"Current definition ({target_def.semantic_name}):\n{cur_json}\n\n"
        f"False Negatives (should match this class, but were missed):\n{fn_block}\n\n"
        f"False Positives (should NOT match this class, but were predicted as it):\n{fp_block}\n\n"
        f"Revise the ClassDefinition so a zero-shot classifier accepts the False Negatives "
        f"and rejects the False Positives.\n"
        f"CONSTRAINTS:\n"
        f"1. Keep it 100% SELF-SUFFICIENT in isolation.\n"
        f"2. NEVER mention other class names, internal labels, or 'other classes'.\n"
        f"3. Generalize the intent boundary; do not overfit to specific proper nouns."
    )
    raw = optimizer_provider.complete_json(
        prompt,
        ClassDefinitionSchema.model_json_schema(),
        temperature=0.3,
    )
    parsed = ClassDefinitionSchema.model_validate(raw)
    return ClassDefinition(
        label=target_def.label,
        semantic_name=parsed.semantic_name,
        definition=parsed.definition,
        inclusion_criteria=parsed.inclusion_criteria,
        exclusion_criteria=parsed.exclusion_criteria,
        representative_patterns=parsed.representative_patterns,
    )


def optimize_definitions_reflective(
    *,
    definitions: dict[str, ClassDefinition],
    dev_texts: list[str],
    dev_labels: list[str],
    evaluate_fn: Callable[[dict[str, ClassDefinition], list[str]], list[str]],
    optimizer_provider: LLMProvider,
    iterations: int = 3,
    top_k_classes: int = 4,
    max_workers: int = 8,
    min_examples_per_class: float = 2.0,
    force: bool = False,
) -> tuple[dict[str, ClassDefinition], OptimizationReport]:
    """Run iterative reflective definition optimization with a sparse-dev overfitting guardrail."""
    num_classes = len(definitions)
    density = len(dev_texts) / max(num_classes, 1)
    if density < min_examples_per_class:
        msg = (
            f"Sparse dev set detected ({len(dev_texts)} examples / {num_classes} classes = "
            f"{density:.2f}/class < {min_examples_per_class:.1f}). Reflective optimization risks "
            f"overfitting."
        )
        if not force:
            raise ValueError(f"{msg} Pass force=True to run anyway.")
        warnings.warn(msg, UserWarning, stacklevel=2)

    best_defs = dict(definitions)
    preds = evaluate_fn(best_defs, dev_texts)
    best_acc, best_f1 = _accuracy_and_macro_f1(dev_labels, preds)
    init_acc, init_f1 = best_acc, best_f1

    errors = [
        (text, truth, pred)
        for text, truth, pred in zip(dev_texts, dev_labels, preds)
        if truth != pred
    ]
    history = [
        OptimizationStep(
            iteration=0,
            dev_accuracy=best_acc,
            dev_macro_f1=best_f1,
            mutated_labels=[],
            accepted=True,
        )
    ]

    for it in range(1, iterations + 1):
        if not errors:
            break

        fn_by_label: dict[str, list[tuple[str, str]]] = defaultdict(list)
        fp_by_label: dict[str, list[tuple[str, str]]] = defaultdict(list)
        err_counts: Counter[str] = Counter()
        for text, truth, pred in errors:
            fn_by_label[truth].append((text, pred))
            fp_by_label[pred].append((text, truth))
            err_counts[truth] += 1
            err_counts[pred] += 1

        targets = [lbl for lbl, _ in err_counts.most_common(top_k_classes) if lbl in best_defs]
        if not targets:
            break

        candidate_defs = dict(best_defs)
        with ThreadPoolExecutor(max_workers=min(len(targets), max_workers)) as pool:
            futs = {
                pool.submit(
                    _reflect_and_refine,
                    optimizer_provider,
                    best_defs[lbl],
                    best_defs,
                    fn_by_label[lbl],
                    fp_by_label[lbl],
                ): lbl
                for lbl in targets
            }
            for fut in as_completed(futs):
                lbl = futs[fut]
                candidate_defs[lbl] = fut.result()

        cand_preds = evaluate_fn(candidate_defs, dev_texts)
        cand_acc, cand_f1 = _accuracy_and_macro_f1(dev_labels, cand_preds)
        improved = (cand_f1 > best_f1) or (cand_f1 == best_f1 and cand_acc > best_acc)

        history.append(
            OptimizationStep(
                iteration=it,
                dev_accuracy=cand_acc,
                dev_macro_f1=cand_f1,
                mutated_labels=targets,
                accepted=improved,
            )
        )
        if improved:
            best_defs = candidate_defs
            best_acc = cand_acc
            best_f1 = cand_f1
            errors = [
                (text, truth, pred)
                for text, truth, pred in zip(dev_texts, dev_labels, cand_preds)
                if truth != pred
            ]

    return best_defs, OptimizationReport(
        initial_accuracy=init_acc,
        initial_macro_f1=init_f1,
        best_accuracy=best_acc,
        best_macro_f1=best_f1,
        history=history,
    )


def optimize_instruction_dspy(
    *,
    dev_texts: list[str],
    dev_labels: list[str],
    dspy_lm: Any,
    instruction_seed: str,
    definitions_block: str,
    allowed_labels: list[str],
    num_trials: int = 8,
) -> str:
    """Optional DSPy MIPROv2 wrapper to optimize the task instruction around definitions."""
    try:
        import dspy
    except ImportError as exc:
        raise ImportError(
            "DSPy backend requested, but `dspy` is not installed. "
            "Install with `pip install defclass[dspy]`."
        ) from exc

    class IntentSig(dspy.Signature):
        definitions: str = dspy.InputField()
        allowed_labels: str = dspy.InputField()
        utterance: str = dspy.InputField()
        intent: str = dspy.OutputField()

    program = dspy.Predict(IntentSig)
    program.signature = program.signature.with_instructions(instruction_seed)

    trainset = [
        dspy.Example(
            definitions=definitions_block,
            allowed_labels=", ".join(allowed_labels),
            utterance=t,
            intent=y,
        ).with_inputs("definitions", "allowed_labels", "utterance")
        for t, y in zip(dev_texts, dev_labels)
    ]

    def _metric(example: Any, pred: Any, trace: Any = None) -> float:
        return float(str(pred.intent).strip() == str(example.intent).strip())

    with dspy.context(lm=dspy_lm):
        optimizer = dspy.MIPROv2(metric=_metric, auto="light", num_threads=4)
        compiled = optimizer.compile(program, trainset=trainset, num_trials=num_trials)
        return str(compiled.signature.instructions)
