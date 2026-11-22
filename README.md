# `defclass`: Definition-Driven Zero-Shot Text Classification

`defclass` is an LLM-provider-agnostic Python library encapsulating empirically validated best practices for zero-shot generative text classification:

- **Provider-agnostic (Generative + Decision Models)**: Specify any `provider` and `model` alongside your own `api_key` for OpenAI / OpenAI-compatible endpoints (LM Studio, vLLM, Groq, etc.), Google Gemini, Anthropic Claude, LiteLLM, or calibrated decision models (`gemini_decision`, `jev`) that return `confidence` and class `probabilities`.
- **Automatic opaque vs. semantic detection**: Automatically detects numeric/ID/UUID labels (requiring `definitions` or `examples` when labels are opaque) and **opacifies semantic labels by default** (`opacify=True`) to eliminate label-name priors.
- **Self-sufficient definition synthesis & isolation auditing**: Generates structured definitions (`definition`, `inclusion_criteria`, `exclusion_criteria`, `representative_patterns`) when none are provided and audits them in isolation to remove sibling-label leaks.
- **Positional debiasing by default**: Shuffles class order on every prediction (`shuffle_order=True`) to prevent primacy/recency/lost-in-the-middle bias.
- **Optional rationale mode (`rationale=True`)**: Prompts generative LLMs to return a brief 1–2 sentence reasoning blurb (`Prediction.rationale`) explaining why the selected class was chosen.
- **Reflective prompt/definition optimization (`optimize`)**: Error-driven hill-climbing loop with a built-in sparse-dev overfitting guardrail (`min_examples_per_class=2.0`) and optional DSPy `MIPROv2` instruction tuning.
- **Hierarchical two-stage routing (`hierarchical="auto"`)**: Clusters large label spaces (`>40` classes) and uses `top-k` cluster routing before leaf disambiguation.
- **Self-consistency ensembling (`votes=3`)**: Optional majority voting across shuffled passes to eliminate residual sampling non-determinism.

## Research Report

The empirical ablation study, cross-architecture benchmarks (zero-shot LLMs, decision models, SetFit, sentence-transformer heads, and TF-IDF baselines), and literature review motivating `defclass`'s defaults are documented in [docs/research_report.md](docs/research_report.md).

## Best Practice: Provide a Golden Reference Dataset

While `defclass` can synthesize definitions from semantic label names alone, **providing a golden reference dataset (`examples` per class plus a labeled `dev` split for `optimize`) is strongly recommended as best practice**. Without labeled reference examples, there is no way to test definition quality, verify alignment with domain-specific annotation conventions, or catch subtle boundary ambiguities.

## Quick Start

### Generative LLM (specify `provider`, `model`, and `api_key`)

```python
from defclass import ZeroShotClassifier

clf = ZeroShotClassifier(
    labels=["card_arrival", "lost_or_stolen_card", "pending_card_payment"],
    provider="openai",          # or "gemini", "anthropic", "openai_compat", "litellm"
    model="gpt-4.1-mini",       # specify any model supported by your provider
    api_key="your-api-key",
    examples={
        "card_arrival": ["My new card hasn't arrived yet", "When will my card be delivered?"],
        "lost_or_stolen_card": ["I lost my debit card", "Someone stole my card"],
        "pending_card_payment": ["Why is this card charge still pending?"],
    },
    domain_context="Retail banking customer support intents",
    rationale=True,
)

pred = clf.predict("My new debit card still hasn't arrived in the mail.")
print(pred.label)      # -> "card_arrival"
print(pred.rationale)  # -> "The utterance asks about an undelivered new debit card..."
```

### Decision Model (with class probabilities)

> **Note:** Decision models (`gemini_decision`, `jev`) cannot generate class definitions on their own. If you pass only a decision model without `definitions` or a generative `generator_provider`, `defclass` emits a warning and falls back to raw label descriptions (or raises a `ValueError` if labels are opaque).

```python
from defclass import GeminiProvider, ZeroShotClassifier

clf = ZeroShotClassifier(
    labels=["card_arrival", "lost_or_stolen_card"],
    provider="gemini_decision",
    model="gemini-decision-1",
    api_key="your-gemini-api-key",
    generator_provider=GeminiProvider(model="gemini-3.5-flash-lite", api_key="your-gemini-api-key"),
    examples={
        "card_arrival": ["My new card hasn't arrived yet"],
        "lost_or_stolen_card": ["Someone stole my card"],
    },
)

pred = clf.predict("I can't find my card anywhere")
print(pred.label, pred.confidence, pred.probabilities)
```
