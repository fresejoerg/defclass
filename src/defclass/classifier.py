"""Main ZeroShotClassifier API encapsulating SOTA zero-shot text classification practices."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import time
from typing import Any

from .definitions import generate_definitions
from .hierarchy import ClusterNode, build_clusters
from .labels import detect_opaque_labels, make_opaque_mapping, match_allowed_label
from .optimizer import optimize_definitions_reflective, optimize_instruction_dspy
from .providers import DecisionProvider, LLMProvider, create_provider
from .types import ClassDefinition, OptimizationReport, Prediction


class ZeroShotClassifier:
    """Provider-agnostic zero-shot text classifier with SOTA best practices built in.

    Supports both generative `LLMProvider` backends and calibrated `DecisionProvider`
    backends (`gemini_decision`, `jev`).

    Defaults:
    - `opacify=True`: replaces semantic labels with UUIDs (and strips semantic titles)
      to eliminate label-name priors and force definition-level reasoning.
    - `shuffle_order=True`: shuffles class order on every call to eliminate positional bias.
    - `audit_definitions=True`: isolation-audits auto-generated definitions to prevent
      sibling-label leaks.
    """

    def __init__(
        self,
        labels: list[str],
        *,
        provider: LLMProvider | DecisionProvider | str,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        generator_provider: LLMProvider | None = None,
        definitions: dict[str, ClassDefinition | dict[str, Any] | str] | None = None,
        examples: dict[str, list[str]] | None = None,
        domain_context: str | None = None,
        examples_per_class: int = 10,
        audit_definitions: bool = True,
        opacify: bool = True,
        opaque_mode: str = "uuid",
        shuffle_order: bool = True,
        rationale: bool = False,
        votes: int = 1,
        hierarchical: bool | str = "auto",
        hierarchical_threshold: int = 40,
        num_clusters: int | None = None,
        top_k_clusters: int = 2,
        seed: int = 42,
        max_workers: int = 8,
    ) -> None:
        if not labels:
            raise ValueError("`labels` must be non-empty.")
        if votes < 1:
            raise ValueError("`votes` must be >= 1.")

        if isinstance(provider, str):
            self.provider: LLMProvider | DecisionProvider = create_provider(
                provider, model, api_key=api_key, base_url=base_url
            )
        else:
            self.provider = provider

        self.generator_provider: LLMProvider | None = (
            generator_provider
            if generator_provider is not None
            else (self.provider if isinstance(self.provider, LLMProvider) else None)
        )

        self.labels = list(labels)
        self.labels_are_opaque = detect_opaque_labels(self.labels)
        self.opacify = opacify
        self.opaque_mode = opaque_mode
        self.shuffle_order = shuffle_order
        self.rationale = rationale
        self.votes = votes
        self.top_k_clusters = top_k_clusters
        self.max_workers = max_workers
        self._rng = random.Random(seed)
        self.instruction = (
            "Classify the following user utterance into exactly one of the classes "
            "defined below. Reason carefully over each class definition to determine membership."
        )

        # Resolve or synthesize self-sufficient class definitions
        if definitions is not None:
            self.definitions = self._normalize_definitions(definitions)
        else:
            if self.labels_are_opaque and not examples:
                raise ValueError(
                    "Detected opaque class labels without `definitions` or `examples`. "
                    "Provide either `definitions` or representative `examples` so self-sufficient "
                    "definitions can be generated."
                )
            if self.generator_provider is None:
                raise ValueError(
                    "A generative `generator_provider` is required to auto-generate definitions "
                    "when using a DecisionProvider without pre-supplied `definitions`."
                )
            self.definitions = generate_definitions(
                provider=self.generator_provider,
                labels=self.labels,
                examples=examples,
                domain_context=domain_context,
                examples_per_class=examples_per_class,
                audit=audit_definitions,
                max_workers=max_workers,
                seed=seed,
            )

        # Build opaque display mapping (always opaque if source labels are opaque or opacify=True)
        self._use_opaque_ids = self.opacify or self.labels_are_opaque
        if self._use_opaque_ids:
            self._raw_to_display = make_opaque_mapping(self.labels, mode=self.opaque_mode)
        else:
            self._raw_to_display = {lbl: lbl for lbl in self.labels}

        # Configure optional two-stage hierarchical routing for large label spaces
        use_hier = (
            len(self.labels) > hierarchical_threshold
            if hierarchical == "auto"
            else bool(hierarchical)
        )
        if use_hier:
            if self.generator_provider is None:
                raise ValueError(
                    "A generative `generator_provider` is required to summarize clusters "
                    "when `hierarchical=True` with a DecisionProvider."
                )
            self._clusters: dict[str, ClusterNode] | None = build_clusters(
                self.generator_provider,
                self.definitions,
                num_clusters=num_clusters,
                max_workers=max_workers,
            )
        else:
            self._clusters = None

    def _normalize_definitions(
        self,
        raw_defs: dict[str, ClassDefinition | dict[str, Any] | str],
    ) -> dict[str, ClassDefinition]:
        normalized: dict[str, ClassDefinition] = {}
        for lbl in self.labels:
            if lbl not in raw_defs:
                raise KeyError(f"Missing definition for label {lbl!r}.")
            val = raw_defs[lbl]
            if isinstance(val, ClassDefinition):
                normalized[lbl] = val
            elif isinstance(val, dict):
                normalized[lbl] = ClassDefinition.from_dict(lbl, val)
            elif isinstance(val, str):
                normalized[lbl] = ClassDefinition(
                    label=lbl,
                    semantic_name=lbl,
                    definition=val,
                    inclusion_criteria=[],
                    exclusion_criteria=[],
                    representative_patterns=[],
                )
            else:
                raise TypeError(f"Unsupported definition type for {lbl!r}: {type(val)}")
        return normalized

    def _route_clusters(self, text: str) -> tuple[list[str], int]:
        assert self._clusters is not None
        cids = list(self._clusters.keys())
        if self.shuffle_order:
            self._rng.shuffle(cids)
        top_m = min(self.top_k_clusters, len(cids))

        if isinstance(self.provider, DecisionProvider):
            criteria = {cid: self._clusters[cid].summary_text for cid in cids}
            res = self.provider.decide(
                text=text,
                instructions="Select the intent cluster that best matches this utterance.",
                criteria=criteria,
            )
            if res.probabilities:
                ranked = [
                    cid for cid, _ in sorted(res.probabilities.items(), key=lambda kv: kv[1], reverse=True)
                ]
            else:
                ranked = [match_allowed_label(res.choice, cids)]
            return ranked[:top_m], sum(len(v) for v in criteria.values())

        block = "\n\n---\n\n".join(self._clusters[cid].summary_text for cid in cids)
        prompt = (
            f"Select the top {top_m} most likely clusters (ordered most to least likely) "
            f"that contain the user utterance's class.\n\n"
            f"{block}\n\n"
            f"Allowed cluster IDs: {', '.join(cids)}\n\n"
            f"Utterance: {text}\n\n"
            f'Respond with JSON: {{"ranked_clusters": ["<cluster_id>", ...]}}'
        )
        schema = {
            "type": "object",
            "properties": {
                "ranked_clusters": {
                    "type": "array",
                    "items": {"type": "string", "enum": cids},
                    "minItems": top_m,
                    "maxItems": top_m,
                }
            },
            "required": ["ranked_clusters"],
        }
        res = self.provider.complete_json(prompt, schema, temperature=0.0)
        chosen: list[str] = []
        for cid in res.get("ranked_clusters", []):
            matched = match_allowed_label(str(cid), cids)
            if matched in self._clusters and matched not in chosen:
                chosen.append(matched)
        return chosen[:top_m] or cids[:top_m], len(prompt)

    def _single_pass_classify(
        self,
        text: str,
        candidate_labels: list[str],
        *,
        use_rationale: bool = False,
        defs_override: dict[str, ClassDefinition] | None = None,
    ) -> tuple[str, float | None, dict[str, float] | None, str | None, int]:
        defs = defs_override or self.definitions
        include_sem = not self._use_opaque_ids

        items: list[tuple[str, str, str]] = []
        for raw_lbl in candidate_labels:
            display = self._raw_to_display[raw_lbl]
            body = defs[raw_lbl].render(display, include_semantic_name=include_sem)
            items.append((display, raw_lbl, body))

        if self.shuffle_order:
            self._rng.shuffle(items)

        display_labels = [d for d, _, _ in items]
        display_to_raw = {d: r for d, r, _ in items}

        if isinstance(self.provider, DecisionProvider):
            criteria = {d: b for d, _, b in items}
            res = self.provider.decide(
                text=text,
                instructions=self.instruction,
                criteria=criteria,
            )
            chosen_display = match_allowed_label(res.choice, display_labels)
            chosen_raw = display_to_raw.get(chosen_display, chosen_display)
            mapped_probs = (
                {
                    display_to_raw.get(match_allowed_label(k, display_labels), k): v
                    for k, v in res.probabilities.items()
                }
                if res.probabilities
                else None
            )
            char_count = sum(len(v) for v in criteria.values()) + len(text)
            return chosen_raw, res.confidence, mapped_probs, None, char_count

        defs_block = "\n\n---\n\n".join(b for _, _, b in items)
        if use_rationale:
            prompt = (
                f"{self.instruction}\n\n"
                f"{defs_block}\n\n"
                f"Allowed class labels: {', '.join(display_labels)}\n\n"
                f"Utterance: {text}\n\n"
                f'Respond with JSON containing a brief 1-2 sentence "rationale" explaining why '
                f'the utterance matches the chosen class definition, followed by "intent": '
                f'{{"rationale": "<brief explanation>", "intent": "<one_allowed_class_label>"}}'
            )
            schema: dict[str, Any] = {
                "type": "object",
                "properties": {
                    "rationale": {
                        "type": "string",
                        "description": "Brief 1-2 sentence explanation of why the selected class was chosen.",
                    },
                    "intent": {"type": "string", "enum": display_labels},
                },
                "required": ["rationale", "intent"],
            }
        else:
            prompt = (
                f"{self.instruction}\n\n"
                f"{defs_block}\n\n"
                f"Allowed class labels: {', '.join(display_labels)}\n\n"
                f"Utterance: {text}\n\n"
                f'Respond with JSON: {{"intent": "<one_allowed_class_label>"}}'
            )
            schema = {
                "type": "object",
                "properties": {"intent": {"type": "string", "enum": display_labels}},
                "required": ["intent"],
            }

        res = self.provider.complete_json(prompt, schema, temperature=0.0)
        chosen_display = match_allowed_label(str(res["intent"]), display_labels)
        chosen_raw = display_to_raw.get(chosen_display, chosen_display)
        rationale_text = str(res["rationale"]).strip() if use_rationale and "rationale" in res else None
        return chosen_raw, None, None, rationale_text, len(prompt)

    def predict(self, text: str, *, rationale: bool | None = None) -> Prediction:
        """Classify a single utterance (supporting decision probabilities and optional rationale)."""
        t0 = time.perf_counter()
        total_chars = 0
        routed_clusters: list[str] | None = None
        use_rationale = self.rationale if rationale is None else rationale

        if self._clusters is not None:
            routed_clusters, c_route = self._route_clusters(text)
            total_chars += c_route
            candidates = [
                lbl for cid in routed_clusters for lbl in self._clusters[cid].members
            ]
        else:
            candidates = list(self.labels)

        votes_list: list[str] = []
        confidences: list[float] = []
        prob_accum: dict[str, float] = defaultdict(float)
        prob_votes = 0
        rationales_by_label: dict[str, str] = {}

        for _ in range(self.votes):
            lbl, conf, probs, rat, c_pass = self._single_pass_classify(
                text,
                candidates,
                use_rationale=use_rationale,
            )
            votes_list.append(lbl)
            total_chars += c_pass
            if conf is not None:
                confidences.append(conf)
            if probs is not None:
                prob_votes += 1
                for k, v in probs.items():
                    prob_accum[k] += v
            if rat is not None and lbl not in rationales_by_label:
                rationales_by_label[lbl] = rat

        final_label = Counter(votes_list).most_common(1)[0][0]
        avg_probs = (
            {k: v / prob_votes for k, v in prob_accum.items()}
            if prob_votes > 0
            else None
        )
        avg_conf = (sum(confidences) / len(confidences)) if confidences else None
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return Prediction(
            label=final_label,
            confidence=avg_conf,
            probabilities=avg_probs,
            rationale=rationales_by_label.get(final_label),
            votes=votes_list,
            routed_clusters=routed_clusters,
            latency_ms=elapsed_ms,
            prompt_chars=total_chars,
        )

    def predict_batch(
        self,
        texts: list[str],
        *,
        rationale: bool | None = None,
        max_workers: int | None = None,
    ) -> list[Prediction]:
        """Classify a batch of utterances concurrently."""
        workers = max_workers or self.max_workers
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda t: self.predict(t, rationale=rationale), texts))

    def optimize(
        self,
        dev_texts: list[str],
        dev_labels: list[str],
        *,
        optimizer_provider: LLMProvider | None = None,
        iterations: int = 3,
        top_k_classes: int = 4,
        min_examples_per_class: float = 2.0,
        force: bool = False,
        dspy_lm: Any = None,
    ) -> OptimizationReport:
        """Optimize class definitions (and optionally instruction via DSPy) on a dev set."""
        opt_prov = optimizer_provider or self.generator_provider
        if opt_prov is None:
            raise ValueError(
                "A generative `optimizer_provider` is required when optimizing definitions "
                "for a DecisionProvider."
            )

        def _eval_fn(cand_defs: dict[str, ClassDefinition], texts: list[str]) -> list[str]:
            def _one(t: str) -> str:
                lbl, _, _, _, _ = self._single_pass_classify(
                    t, self.labels, defs_override=cand_defs
                )
                return lbl

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                return list(pool.map(_one, texts))

        best_defs, report = optimize_definitions_reflective(
            definitions=self.definitions,
            dev_texts=dev_texts,
            dev_labels=dev_labels,
            evaluate_fn=_eval_fn,
            optimizer_provider=opt_prov,
            iterations=iterations,
            top_k_classes=top_k_classes,
            max_workers=self.max_workers,
            min_examples_per_class=min_examples_per_class,
            force=force,
        )
        self.definitions = best_defs

        if dspy_lm is not None:
            defs_block = "\n\n---\n\n".join(
                self.definitions[lbl].render(
                    self._raw_to_display[lbl], include_semantic_name=not self._use_opaque_ids
                )
                for lbl in self.labels
            )
            display_labels = [self._raw_to_display[lbl] for lbl in self.labels]
            mapped_dev_labels = [self._raw_to_display[y] for y in dev_labels]
            self.instruction = optimize_instruction_dspy(
                dev_texts=dev_texts,
                dev_labels=mapped_dev_labels,
                dspy_lm=dspy_lm,
                instruction_seed=self.instruction,
                definitions_block=defs_block,
                allowed_labels=display_labels,
            )

        return report

    def save_definitions(self, path: str | Path) -> None:
        """Persist current definitions to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {lbl: self.definitions[lbl].to_dict() for lbl in self.labels}
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load_definitions(cls, path: str | Path) -> dict[str, ClassDefinition]:
        """Load definitions saved by `save_definitions`."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return {lbl: ClassDefinition.from_dict(lbl, data) for lbl, data in raw.items()}
