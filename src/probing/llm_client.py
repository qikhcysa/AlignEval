"""LLM client: supports OpenAI-compatible API and mock mode."""
from __future__ import annotations

import logging
import random
import time
from typing import Any

from src.models import ProbePrompt, ProbeResult

logger = logging.getLogger(__name__)

# Mock response templates that simulate an LLM's answer
_MOCK_FACTUAL_RESPONSES = [
    "{entity} is a fundamental concept in {domain}. It refers to a specific process or object "
    "that plays an important role in the field. The key properties include being well-defined, "
    "measurable, and having documented relationships with other concepts.",
    "{entity} can be described as an important term in {domain}. It is associated with several "
    "other concepts and has been studied extensively. Its main characteristics include relevance "
    "to the core domain questions and interaction with related entities.",
]

_MOCK_RELATIONAL_RESPONSES = [
    "{entity} is directly related to {related_entity}. The relationship involves {entity} "
    "having a causal or structural connection to {related_entity}. Specifically, {entity} "
    "influences {related_entity} through a well-documented mechanism.",
    "The relationship between {entity} and {related_entity} is characterized by a strong "
    "dependency. {entity} typically precedes or enables {related_entity} in the domain context. "
    "This interaction is well established in the literature.",
]

_MOCK_REVERSE_RESPONSES = [
    "Given the relationship between {entity} and {related_entity}, the underlying mechanism "
    "involves {entity} acting as a precursor or catalyst. If {entity} were removed, {related_entity} "
    "would be significantly impacted. The preconditions require both entities to be present.",
    "Reasoning backwards from {related_entity}: the root cause traces back to {entity}. "
    "The mechanism involves {entity} triggering a series of events that ultimately affect "
    "{related_entity}. This chain is well documented in domain research.",
]


def _mock_response(prompt: ProbePrompt) -> str:
    """Generate a realistic-looking mock LLM response."""
    from src.models import ProbeLevel
    templates = {
        ProbeLevel.FACTUAL: _MOCK_FACTUAL_RESPONSES,
        ProbeLevel.RELATIONAL: _MOCK_RELATIONAL_RESPONSES,
        ProbeLevel.REVERSE: _MOCK_REVERSE_RESPONSES,
    }
    tmpl_list = templates.get(prompt.level, _MOCK_FACTUAL_RESPONSES)
    tmpl = random.choice(tmpl_list)
    return tmpl.format(
        entity=prompt.entity,
        related_entity=prompt.related_entity or "the related concept",
        domain="the domain",
        relation=prompt.expected_relation.replace("_", " ") if prompt.expected_relation else "related to",
    )


class LLMClient:
    """Client for querying an OpenAI-compatible LLM API (with mock fallback)."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        mock_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.mock_mode = mock_mode or not api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Any = None

        if not self.mock_mode:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key, base_url=base_url)
            except Exception as exc:
                logger.warning("Failed to initialize OpenAI client: %s. Using mock mode.", exc)
                self.mock_mode = True

    def query(self, prompt: ProbePrompt) -> ProbeResult:
        """Send a probe prompt to the LLM and return a ProbeResult."""
        start = time.monotonic()

        if self.mock_mode:
            response_text = _mock_response(prompt)
            latency = (time.monotonic() - start) * 1000
            return ProbeResult(
                prompt=prompt,
                response=response_text,
                model_name="mock",
                latency_ms=latency,
            )

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert assistant. Answer the question precisely, "
                            "mentioning specific entities and their relationships. "
                            "Be factual and concise."
                        ),
                    },
                    {"role": "user", "content": prompt.prompt_text},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            response_text = completion.choices[0].message.content or ""
            latency = (time.monotonic() - start) * 1000
            return ProbeResult(
                prompt=prompt,
                response=response_text,
                model_name=self.model,
                latency_ms=latency,
            )
        except Exception as exc:
            logger.error("LLM query failed: %s", exc)
            response_text = _mock_response(prompt)
            latency = (time.monotonic() - start) * 1000
            return ProbeResult(
                prompt=prompt,
                response=response_text,
                model_name="mock-fallback",
                latency_ms=latency,
            )

    def query_batch(self, prompts: list[ProbePrompt]) -> list[ProbeResult]:
        """Query the LLM for a list of prompts."""
        return [self.query(p) for p in prompts]
