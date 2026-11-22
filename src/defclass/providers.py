"""Provider-agnostic LLM abstraction with user-supplied API key support."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
import time
from typing import Any
import urllib.error
import urllib.request

from .types import DecisionResult


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    return cleaned


class LLMProvider(ABC):
    """Abstract interface for any structured-JSON-capable LLM backend."""

    @abstractmethod
    def complete_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Execute a prompt and return parsed JSON conforming to schema."""


class DecisionProvider(ABC):
    """Abstract interface for calibrated decision-model backends (e.g. Gemini Decision, Jev)."""

    @abstractmethod
    def decide(
        self,
        text: str,
        instructions: str,
        criteria: dict[str, str],
    ) -> DecisionResult:
        """Select one choice from criteria and return choice, confidence, and probabilities."""


class OpenAICompatibleProvider(LLMProvider):
    """Zero-dependency OpenAI-compatible HTTP provider (OpenAI, LM Studio, vLLM, Groq, etc.)."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        max_retries: int = 3,
        use_json_schema: bool = True,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_json_schema = use_json_schema

    def complete_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if self.use_json_schema:
            strict_schema = dict(schema)
            strict_schema.setdefault("additionalProperties", False)
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": strict_schema,
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=body,
                    headers=headers,
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                content = str(data["choices"][0]["message"]["content"])
                return json.loads(_strip_code_fences(content))
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"OpenAICompatibleProvider failed after {self.max_retries} retries: {last_err}")


class GeminiProvider(LLMProvider):
    """Google Gemini provider via google-genai (user-supplied api_key or GEMINI_API_KEY)."""

    def __init__(
        self,
        model: str = "gemini-3.5-flash-lite",
        api_key: str | None = None,
        max_retries: int = 3,
    ) -> None:
        from google import genai

        self.model = model
        self.max_retries = max_retries
        key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = genai.Client(api_key=key) if key else genai.Client()

    def complete_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                interaction = self._client.interactions.create(
                    model=self.model,
                    input=prompt,
                    generation_config={"temperature": temperature},
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": schema,
                    },
                )
                return json.loads(_strip_code_fences(interaction.output_text))
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"GeminiProvider failed after {self.max_retries} retries: {last_err}")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider using forced tool-use for guaranteed JSON schema output."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 1024,
        max_retries: int = 3,
    ) -> None:
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()

    def complete_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    tools=[
                        {
                            "name": "emit_result",
                            "description": "Emit the structured classification or definition JSON.",
                            "input_schema": schema,
                        }
                    ],
                    tool_choice={"type": "tool", "name": "emit_result"},
                    messages=[{"role": "user", "content": prompt}],
                )
                for block in msg.content:
                    if getattr(block, "type", None) == "tool_use":
                        return dict(block.input)
                raise ValueError("Anthropic response did not contain a tool_use block.")
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"AnthropicProvider failed after {self.max_retries} retries: {last_err}")


class LiteLLMProvider(LLMProvider):
    """Optional catch-all provider via litellm (100+ providers)."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_retries = max_retries

    def complete_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        import litellm

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = litellm.completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_output",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                )
                content = str(resp.choices[0].message.content)
                return json.loads(_strip_code_fences(content))
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LiteLLMProvider failed after {self.max_retries} retries: {last_err}")


class GeminiDecisionProvider(DecisionProvider):
    """Google Gemini Decisions API provider (gemini-decision-1)."""

    def __init__(
        self,
        model: str = "gemini-decision-1",
        api_key: str | None = None,
        max_retries: int = 3,
    ) -> None:
        from google import genai

        self.model = model
        self.max_retries = max_retries
        key = api_key or os.environ.get("GEMINI_API_KEY")
        kwargs: dict[str, Any] = {"http_options": {"api_version": "v1beta"}}
        if key:
            kwargs["api_key"] = key
        self._client = genai.Client(**kwargs)

    def decide(
        self,
        text: str,
        instructions: str,
        criteria: dict[str, str],
    ) -> DecisionResult:
        if len(criteria) > 255:
            raise ValueError(
                f"Gemini Decisions API supports at most 255 choice options, got {len(criteria)}."
            )
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.decisions.create(
                    model=self.model,
                    input=text,
                    questions={
                        "intent": {
                            "type": "choice",
                            "instructions": instructions,
                            "criteria": criteria,
                        }
                    },
                )
                ans = response.answers["intent"]
                probs = (
                    {str(k): float(v) for k, v in ans.probabilities.items()}
                    if getattr(ans, "probabilities", None)
                    else None
                )
                conf = (
                    float(ans.confidence)
                    if getattr(ans, "confidence", None) is not None
                    else None
                )
                return DecisionResult(
                    choice=str(ans.choice),
                    confidence=conf,
                    probabilities=probs,
                )
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"GeminiDecisionProvider failed after {self.max_retries} retries: {last_err}")


class JevDecisionProvider(DecisionProvider):
    """TypeSafe AI Jev decision-model provider."""

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 3,
    ) -> None:
        from typesafe_sdk import Choice, TypeSafeClient

        self.max_retries = max_retries
        self._choice_cls = Choice
        key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self._client = TypeSafeClient(api_key=key) if key else TypeSafeClient()

    def decide(
        self,
        text: str,
        instructions: str,
        criteria: dict[str, str],
    ) -> DecisionResult:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.system_one(
                    state={"utterance": text},
                    questions={
                        "intent": self._choice_cls(
                            instructions=instructions,
                            criteria=criteria,
                        )
                    },
                )
                ans = response.answers["intent"]
                probs = (
                    {str(k): float(v) for k, v in ans.probabilities.items()}
                    if getattr(ans, "probabilities", None)
                    else None
                )
                conf = (
                    float(ans.confidence)
                    if getattr(ans, "confidence", None) is not None
                    else None
                )
                return DecisionResult(
                    choice=str(ans.choice),
                    confidence=conf,
                    probabilities=probs,
                )
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"JevDecisionProvider failed after {self.max_retries} retries: {last_err}")


def create_provider(
    provider: str,
    model: str | None = None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMProvider | DecisionProvider:
    """Instantiate an LLMProvider or DecisionProvider by name."""
    key = provider.lower()
    if key == "gemini_decision":
        return GeminiDecisionProvider(model=model or "gemini-decision-1", api_key=api_key)
    if key == "jev":
        return JevDecisionProvider(api_key=api_key)
    if not model:
        raise ValueError(f"`model` is required for generative provider {provider!r}.")
    if key == "openai":
        return OpenAICompatibleProvider(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
        )
    if key in ("openai_compat", "lmstudio", "vllm", "ollama"):
        return OpenAICompatibleProvider(
            model=model,
            api_key=api_key or "local",
            base_url=base_url or "http://127.0.0.1:1234/v1",
        )
    if key == "gemini":
        return GeminiProvider(model=model, api_key=api_key)
    if key == "anthropic":
        return AnthropicProvider(model=model, api_key=api_key)
    if key == "litellm":
        return LiteLLMProvider(model=model, api_key=api_key, api_base=base_url)
    raise ValueError(f"Unknown provider {provider!r}.")
