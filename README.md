# `defclass`: Definition-Driven Zero-Shot Text Classification

`defclass` is an LLM-provider-agnostic Python library encapsulating empirically validated best practices for zero-shot generative text classification:

- **Provider-agnostic**: Built-in adapters for OpenAI / OpenAI-compatible endpoints (LM Studio, vLLM, Groq, etc.), Google Gemini, Anthropic Claude, and LiteLLM, with user-supplied API keys or custom `LLMProvider` implementations.
- **Automatic opaque vs. semantic detection**: Automatically detects numeric/ID/UUID labels (requiring `definitions` or `examples` when labels are opaque) and **opacifies semantic labels by default** (`opacify=True`) to eliminate label-name priors.
- **Self-sufficient definition synthesis & isolation auditing**: Generates structured definitions (`definition`, `inclusion_criteria`, `exclusion_criteria`, `representative_patterns`) when none are provided and audits them in isolation to remove sibling-label leaks.
- **Positional debiasing by default**: Shuffles class order on every prediction (`shuffle_order=True`) to prevent primacy/recency/lost-in-the-middle bias.
- **Reflective prompt/definition optimization (`optimize`)**: Error-driven hill-climbing loop with a built-in sparse-dev overfitting guardrail (`min_examples_per_class=2.0`) and optional DSPy `MIPROv2` instruction tuning.
- **Hierarchical two-stage routing (`hierarchical="auto"`)**: Clusters large label spaces (`>40` classes) and uses `top-k` cluster routing before leaf disambiguation.
- **Self-consistency ensembling (`votes=3`)**: Optional majority voting across shuffled passes to eliminate residual sampling non-determinism.

## Quick Start

```python
from defclass import ZeroShotClassifier

clf = ZeroShotClassifier(
    labels=["card_arrival", "lost_or_stolen_card", "pending_card_payment"],
    provider="openai",          # or "gemini", "anthropic", "openai_compat", "litellm"
    model="gpt-4.1-mini",
    api_key="your-api-key",
    domain_context="Retail banking customer support intents",
)

pred = clf.predict("My new debit card still hasn't arrived in the mail.")
print(pred.label)  # -> "card_arrival"
```
