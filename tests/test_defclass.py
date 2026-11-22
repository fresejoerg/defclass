"""Unit tests for defclass using a deterministic mock LLMProvider."""

from __future__ import annotations

import re
import tempfile
import unittest

from defclass import (
    ClassDefinition,
    DecisionProvider,
    DecisionResult,
    LLMProvider,
    ZeroShotClassifier,
    detect_opaque_labels,
    is_opaque_label,
)


class MockProvider(LLMProvider):
    """Deterministic mock LLMProvider inspecting prompts and schemas."""

    def __init__(self, *, flip_first_vote: bool = False) -> None:
        self.prompts: list[str] = []
        self.flip_first_vote = flip_first_vote
        self._classify_calls = 0

    def complete_json(self, prompt: str, schema: dict, *, temperature: float = 0.0) -> dict:
        self.prompts.append(prompt)
        props = schema.get("properties", {})

        # 1. Definition synthesis
        if "inclusion_criteria" in props and "is_self_sufficient" not in props:
            m = re.search(r"Target class label: '([^']+)'", prompt)
            lbl = m.group(1) if m else "unknown"
            return {
                "semantic_name": f"Intent {lbl}",
                "definition": f"Requests related to {lbl}.",
                "inclusion_criteria": [f"Mentions {lbl} topic"],
                "exclusion_criteria": ["Unrelated requests"],
                "representative_patterns": [f"help with {lbl}"],
            }

        # 2. Isolation audit (repairs if 'LEAK' appears in definition)
        if "is_self_sufficient" in props:
            if "LEAK" in prompt:
                return {
                    "is_self_sufficient": False,
                    "has_external_label_refs": True,
                    "issues": ["Leaked sibling label"],
                    "revised_definition": {
                        "semantic_name": "Repaired Intent",
                        "definition": "Clean self-sufficient definition.",
                        "inclusion_criteria": ["Clean inclusion"],
                        "exclusion_criteria": ["Clean exclusion"],
                        "representative_patterns": ["clean pattern"],
                    },
                }
            return {
                "is_self_sufficient": True,
                "has_external_label_refs": False,
                "issues": [],
                "revised_definition": None,
            }

        # 3. Cluster summary
        if "summary_name" in props:
            return {
                "summary_name": "Cluster Theme",
                "summary_definition": "Covers grouped intents.",
                "key_signals": ["signal A", "signal B"],
            }

        # 4. Stage 1 cluster routing
        if "ranked_clusters" in props:
            cids = props["ranked_clusters"]["items"]["enum"]
            top_m = props["ranked_clusters"]["minItems"]
            return {"ranked_clusters": cids[:top_m]}

        # 5. Classification
        if "intent" in props:
            self._classify_calls += 1
            allowed = props["intent"]["enum"]
            if self.flip_first_vote and self._classify_calls == 1 and len(allowed) > 1:
                return {"intent": allowed[1]}
            # Pick the class whose definition matches utterance keyword
            utt_match = re.search(r"Utterance: (.+)", prompt)
            utt = utt_match.group(1).lower() if utt_match else ""
            chosen_cid = allowed[0]
            for block in prompt.split("\n\n---\n\n"):
                if "Definition:" not in block:
                    continue
                cls_match = re.search(r"Class: ([^\s\n]+)", block)
                if not cls_match:
                    continue
                cid = cls_match.group(1)
                if "weather" in utt and "weather" in block.lower():
                    chosen_cid = cid
                    break
                if "music" in utt and "music" in block.lower():
                    chosen_cid = cid
                    break
            if "rationale" in props:
                return {
                    "rationale": f"The utterance '{utt}' matches the selected class definition.",
                    "intent": chosen_cid,
                }
            return {"intent": chosen_cid}


class MockDecisionProvider(DecisionProvider):
    """Deterministic mock DecisionProvider returning calibrated probabilities."""

    def decide(
        self,
        text: str,
        instructions: str,
        criteria: dict[str, str],
    ) -> DecisionResult:
        keys = list(criteria.keys())
        probs: dict[str, float] = {}
        for k, desc in criteria.items():
            probs[k] = 0.85 if ("weather" in text.lower() and "weather" in desc.lower()) else 0.15
        total = sum(probs.values()) or 1.0
        probs = {k: v / total for k, v in probs.items()}
        best = max(probs, key=lambda k: probs[k]) if keys else ""
        return DecisionResult(choice=best, confidence=probs[best], probabilities=probs)

        raise AssertionError(f"Unexpected schema: {schema}")


class TestDefClass(unittest.TestCase):
    def test_opaque_detection(self) -> None:
        self.assertTrue(is_opaque_label("0"))
        self.assertTrue(is_opaque_label("cls-12"))
        self.assertTrue(is_opaque_label("550e8400-e29b-41d4-a716-446655440000"))
        self.assertFalse(is_opaque_label("card_arrival"))
        self.assertTrue(detect_opaque_labels(["0", "1", "2"]))
        self.assertFalse(detect_opaque_labels(["weather", "music", "flights"]))

    def test_opaque_without_defs_or_examples_raises(self) -> None:
        with self.assertRaises(ValueError):
            ZeroShotClassifier(labels=["0", "1"], provider=MockProvider())

    def test_opacify_and_shuffle_defaults(self) -> None:
        prov = MockProvider()
        clf = ZeroShotClassifier(
            labels=["weather", "music", "flights"],
            provider=prov,
            hierarchical=False,
        )
        # Verify semantic names are stripped & UUIDs used in prompt
        pred1 = clf.predict("play some music")
        pred2 = clf.predict("check the weather")
        self.assertEqual(pred1.label, "music")
        self.assertEqual(pred2.label, "weather")

        cls_prompt = prov.prompts[-1]
        self.assertNotIn("(Intent weather)", cls_prompt)
        self.assertIn("Class: ", cls_prompt)

    def test_isolation_audit_repairs_leaks(self) -> None:
        prov = MockProvider()
        clf = ZeroShotClassifier(
            labels=["weather", "LEAK_class"],
            provider=prov,
            audit_definitions=True,
            hierarchical=False,
        )
        self.assertEqual(clf.definitions["LEAK_class"].semantic_name, "Repaired Intent")

    def test_majority_voting_resolves_transient_flip(self) -> None:
        prov = MockProvider(flip_first_vote=True)
        defs = {
            "weather": "Forecasts and weather conditions.",
            "music": "Playing songs and music.",
        }
        clf = ZeroShotClassifier(
            labels=["weather", "music"],
            provider=prov,
            definitions=defs,
            shuffle_order=False,
            votes=3,
            hierarchical=False,
        )
        pred = clf.predict("check the weather")
        self.assertEqual(pred.label, "weather")
        self.assertEqual(len(pred.votes), 3)

    def test_sparse_dev_guardrail_and_optimization(self) -> None:
        prov = MockProvider()
        defs = {
            "weather": "Forecasts and weather conditions.",
            "music": "Playing songs and music.",
            "flights": "Flight schedules and routes.",
        }
        clf = ZeroShotClassifier(
            labels=["weather", "music", "flights"],
            provider=prov,
            definitions=defs,
            hierarchical=False,
        )
        # 3 examples / 3 classes = 1.0 < 2.0 -> raises without force=True
        with self.assertRaises(ValueError):
            clf.optimize(["w", "m", "f"], ["weather", "music", "flights"])

        # With >=2 examples/class, optimization runs cleanly
        report = clf.optimize(
            ["weather forecast", "rain today", "play music", "song", "book flights", "airline"],
            ["weather", "weather", "music", "music", "flights", "flights"],
            iterations=1,
        )
        self.assertGreaterEqual(report.best_macro_f1, 0.0)

    def test_hierarchical_top2_routing_and_save_load(self) -> None:
        prov = MockProvider()
        defs = {
            "weather": "Forecasts and weather conditions.",
            "music": "Playing songs and music.",
            "flights": "Flight schedules and routes.",
            "dining": "Restaurant reservations.",
        }
        clf = ZeroShotClassifier(
            labels=list(defs.keys()),
            provider=prov,
            definitions=defs,
            hierarchical=True,
            num_clusters=2,
            top_k_clusters=2,
        )
        pred = clf.predict("check the weather")
        self.assertEqual(pred.label, "weather")
        self.assertIsNotNone(pred.routed_clusters)
        self.assertEqual(len(pred.routed_clusters), 2)

        with tempfile.TemporaryDirectory() as tmp:
            p = f"{tmp}/defs.json"
            clf.save_definitions(p)
            loaded = ZeroShotClassifier.load_definitions(p)
            self.assertEqual(set(loaded.keys()), set(defs.keys()))
            self.assertIsInstance(loaded["weather"], ClassDefinition)

    def test_rationale_mode(self) -> None:
        prov = MockProvider()
        defs = {
            "weather": "Forecasts and weather conditions.",
            "music": "Playing songs and music.",
        }
        clf = ZeroShotClassifier(
            labels=["weather", "music"],
            provider=prov,
            definitions=defs,
            rationale=True,
            hierarchical=False,
        )
        pred = clf.predict("check the weather")
        self.assertEqual(pred.label, "weather")
        self.assertIsNotNone(pred.rationale)
        self.assertIn("weather", pred.rationale.lower())

        pred_no_rat = clf.predict("play music", rationale=False)
        self.assertEqual(pred_no_rat.label, "music")
        self.assertIsNone(pred_no_rat.rationale)

    def test_decision_provider_probabilities_and_hierarchy(self) -> None:
        dec = MockDecisionProvider()
        gen = MockProvider()
        defs = {
            "weather": "Forecasts and weather conditions.",
            "music": "Playing songs and music.",
            "flights": "Flight schedules and routes.",
            "dining": "Restaurant reservations.",
        }
        clf = ZeroShotClassifier(
            labels=list(defs.keys()),
            provider=dec,
            generator_provider=gen,
            definitions=defs,
            hierarchical=True,
            num_clusters=2,
            top_k_clusters=2,
        )
        pred = clf.predict("check the weather")
        self.assertEqual(pred.label, "weather")
        self.assertIsNotNone(pred.confidence)
        self.assertIsNotNone(pred.probabilities)
        self.assertIn("weather", pred.probabilities)
        self.assertAlmostEqual(sum(pred.probabilities.values()), 1.0, places=5)
        self.assertGreater(pred.probabilities["weather"], pred.probabilities["music"])

    def test_decision_provider_only_warns_without_definitions(self) -> None:
        dec = MockDecisionProvider()
        with self.assertWarns(UserWarning):
            clf = ZeroShotClassifier(
                labels=["weather", "music"],
                provider=dec,
                hierarchical=False,
            )
        pred = clf.predict("check the weather")
        self.assertEqual(pred.label, "weather")

        with self.assertRaises(ValueError):
            ZeroShotClassifier(labels=["0", "1"], provider=dec)


if __name__ == "__main__":
    unittest.main()
