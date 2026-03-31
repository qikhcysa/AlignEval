"""Local HuggingFace model prober: drives fine-tuned models as a drop-in for LLMClient."""
from __future__ import annotations

import logging
import time
from typing import Any

from src.models import ProbePrompt, ProbeResult
from src.probing.llm_client import _mock_response

logger = logging.getLogger(__name__)


class ModelProber:
    """Probe a local HuggingFace causal-LM model with ProbePrompts.

    This is the local-model equivalent of :class:`~src.probing.LLMClient`.
    It accepts either a pre-loaded ``(model, tokenizer)`` pair or a Hugging
    Face model identifier / local checkpoint path to be loaded lazily on the
    first ``query`` call.

    When *mock_mode* is ``True`` (or when the ``transformers`` package is not
    installed), the class falls back to the same mock responses used by
    :class:`~src.probing.LLMClient`, so the full evaluation pipeline can be
    exercised without any GPU or model weights.

    Parameters
    ----------
    model_name_or_path:
        A Hugging Face Hub id (e.g. ``"gpt2"``) or local checkpoint directory.
        Ignored when *model* is provided directly.
    model:
        A pre-loaded ``transformers`` model object.  Mutually exclusive with
        *model_name_or_path*.
    tokenizer:
        A pre-loaded tokenizer matching *model*.
    mock_mode:
        Force mock responses regardless of whether a real model is available.
    max_new_tokens:
        Maximum number of new tokens generated per response.
    device:
        Device string passed to the ``transformers`` text-generation pipeline
        (e.g. ``"cpu"``, ``"cuda"``).  Defaults to ``"cpu"``.
    """

    def __init__(
        self,
        model_name_or_path: str = "",
        model: Any = None,
        tokenizer: Any = None,
        mock_mode: bool = False,
        max_new_tokens: int = 128,
        device: str = "cpu",
    ):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._model = model
        self._tokenizer = tokenizer
        self._pipeline: Any = None

        try:
            import transformers  # noqa: F401
            self._has_transformers = True
        except ImportError:
            self._has_transformers = False
            if not mock_mode:
                logger.warning(
                    "transformers is not installed; ModelProber will run in mock mode. "
                    "Install it with: pip install transformers torch"
                )

        # Enable mock when explicitly requested, when transformers is absent,
        # or when neither a path nor a pre-loaded model has been supplied.
        self.mock_mode = mock_mode or not self._has_transformers or (
            not model_name_or_path and model is None
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pipeline(self) -> Any:
        """Lazily build a ``text-generation`` pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        from transformers import pipeline as hf_pipeline

        if self._model is not None and self._tokenizer is not None:
            self._pipeline = hf_pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                device=self.device,
            )
        else:
            self._pipeline = hf_pipeline(
                "text-generation",
                model=self.model_name_or_path,
                device=self.device,
            )
        return self._pipeline

    # ------------------------------------------------------------------
    # Public API  (mirrors LLMClient)
    # ------------------------------------------------------------------

    def query(self, prompt: ProbePrompt) -> ProbeResult:
        """Generate a response for *prompt* and return a :class:`~src.models.ProbeResult`."""
        start = time.monotonic()

        if self.mock_mode:
            response_text = _mock_response(prompt)
            latency = (time.monotonic() - start) * 1000
            return ProbeResult(
                prompt=prompt,
                response=response_text,
                model_name="mock-local",
                latency_ms=latency,
            )

        try:
            pipe = self._get_pipeline()
            outputs = pipe(
                prompt.prompt_text,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=pipe.tokenizer.eos_token_id,
            )
            # The pipeline returns the full text including the prompt prefix;
            # strip it so only the generated continuation is stored.
            full_text: str = outputs[0]["generated_text"]
            response_text = full_text[len(prompt.prompt_text):].strip()
            latency = (time.monotonic() - start) * 1000
            model_label = self.model_name_or_path or "local-model"
            return ProbeResult(
                prompt=prompt,
                response=response_text,
                model_name=model_label,
                latency_ms=latency,
            )
        except Exception as exc:
            logger.error("ModelProber.query failed: %s", exc)
            response_text = _mock_response(prompt)
            latency = (time.monotonic() - start) * 1000
            return ProbeResult(
                prompt=prompt,
                response=response_text,
                model_name="mock-fallback",
                latency_ms=latency,
            )

    def query_batch(self, prompts: list[ProbePrompt]) -> list[ProbeResult]:
        """Query the model for a list of prompts."""
        return [self.query(p) for p in prompts]
