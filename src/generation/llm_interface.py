"""
Module 7 — LLMInterface
Single-responsibility LLM call wrapper for the Medical Knowledge Assistant.

Responsibilities
----------------
* Accept an assembled prompt (flat str OR OpenAI messages list) from PromptBuilder
* Call GPT-4o via the OpenAI Chat Completions API (primary path)
* Fall back to Llama-3-8B via Ollama if OpenAI is unavailable or raises
* Parse the structured JSON response into a MedicalAnswer object
* Derive confidence from graph_subgraph.path_confidence (product of edge
  confidences along the retrieved path) — overrides LLM-reported confidence
* Always populate the medical safety disclaimer in the returned answer

Environment variables (loaded from .env)
-----------------------------------------
OPENAI_API_KEY      OpenAI API key
OPENAI_MODEL        Model name (default: gpt-4o)
OLLAMA_BASE_URL     Ollama server base URL (default: http://localhost:11434)
OLLAMA_MODEL        Ollama model name (default: llama3:8b)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

from src.utils.models import (
    Citation,
    MedicalAnswer,
    RetrievalMode,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OpenAI import
# ---------------------------------------------------------------------------

try:
    import openai as openai_lib
    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    openai_lib = None  # type: ignore[assignment]
    OPENAI_AVAILABLE = False
    logger.warning("openai package not installed — LLMInterface will use Ollama only")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_OPENAI_MODEL: str = "gpt-4o"
_DEFAULT_OLLAMA_MODEL: str = "llama3:8b"
_DEFAULT_OLLAMA_BASE: str = "http://localhost:11434"
_DEFAULT_MAX_TOKENS: int = 2000
_DEFAULT_TEMPERATURE: float = 0.1  # low for factual, reproducible medical answers

_FALLBACK_DISCLAIMER: str = (
    "⚠️ This information is for educational purposes only and does not "
    "constitute medical advice. Always consult a qualified healthcare "
    "professional before making any medical decisions."
)

_FALLBACK_ANSWER_TEXT: str = (
    "I was unable to generate a structured answer due to a technical error. "
    "Please try again or consult a qualified healthcare professional."
)


# ---------------------------------------------------------------------------
# LLMInterface
# ---------------------------------------------------------------------------


class LLMInterface:
    """Thin wrapper that calls GPT-4o (or Ollama) and returns a MedicalAnswer.

    Clients are initialised lazily on first use so the object can be
    constructed even when the respective API keys are not yet available
    (useful for testing).

    Parameters
    ----------
    openai_api_key:
        OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
    openai_model:
        OpenAI model name. Falls back to ``OPENAI_MODEL`` env var or
        ``"gpt-4o"``.
    ollama_base_url:
        Ollama server base URL. Falls back to ``OLLAMA_BASE_URL`` env var
        or ``"http://localhost:11434"``.
    ollama_model:
        Ollama model name. Falls back to ``OLLAMA_MODEL`` env var or
        ``"llama3:8b"``.
    max_tokens:
        Maximum tokens in the LLM completion (default 2 000).
    temperature:
        Sampling temperature (default 0.1 — low for factual answers).

    Example
    -------
    >>> llm = LLMInterface()
    >>> messages = prompt_builder.build_messages(query, subgraph, chunks, snippets, merged)
    >>> answer = llm.call_llm(messages, retrieval_result)
    >>> print(answer.answer)
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
    ) -> None:
        # Use `is not None` (not `or`) so that passing "" explicitly
        # means "no key / disable OpenAI" without silently falling back
        # to whatever is in the environment.
        self.openai_api_key: str = (
            openai_api_key if openai_api_key is not None
            else os.getenv("OPENAI_API_KEY", "")
        )
        self.openai_model: str = (
            openai_model or os.getenv("OPENAI_MODEL", _DEFAULT_OPENAI_MODEL)
        )
        self.ollama_base_url: str = (
            ollama_base_url or os.getenv("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_BASE)
        ).rstrip("/")
        self.ollama_model: str = (
            ollama_model or os.getenv("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL)
        )
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Clients are lazily initialised — None until first use
        self._openai_client: Optional[Any] = None
        self._ollama_client: Optional[Any] = None

        logger.info(
            "LLMInterface ready (openai_model=%s, ollama_model=%s, "
            "openai_key=%s, ollama_url=%s)",
            self.openai_model,
            self.ollama_model,
            "set" if self.openai_api_key else "missing",
            self.ollama_base_url,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call_llm(
        self,
        prompt: Union[str, List[dict]],
        retrieval_result: Optional[Any] = None,
    ) -> MedicalAnswer:
        """Send a prompt to the LLM and return a parsed MedicalAnswer.

        Tries GPT-4o first; falls back to Ollama on any failure.  If both
        fail a graceful error ``MedicalAnswer`` is returned instead of
        raising.

        Parameters
        ----------
        prompt:
            Either a flat string (from ``PromptBuilder.build()``) or an
            OpenAI-style messages list (from ``PromptBuilder.build_messages()``).
        retrieval_result:
            Optional ``RetrievalResult`` from the orchestrator.  When present
            its ``graph_subgraph.path_confidence`` overrides the LLM-reported
            confidence and its ``retrieval_mode`` is propagated to the answer.

        Returns
        -------
        MedicalAnswer
            Always returns a valid object — never raises.
        """
        messages = self._to_messages(prompt)
        raw_content: Optional[str] = None
        provider_used: str = "none"

        # --- Primary: OpenAI GPT-4o ---
        if self.openai_api_key:
            try:
                raw_content = self._call_openai(messages)
                provider_used = "openai"
                logger.info("LLMInterface: response from OpenAI (%d chars)", len(raw_content))
            except Exception as exc:  # noqa: BLE001
                logger.warning("OpenAI call failed (%s) — trying Ollama fallback", exc)

        # --- Fallback: Ollama ---
        if raw_content is None:
            try:
                raw_content = self._call_ollama(messages)
                provider_used = "ollama"
                logger.info("LLMInterface: response from Ollama (%d chars)", len(raw_content))
            except Exception as exc:  # noqa: BLE001
                logger.error("Ollama fallback also failed: %s", exc)

        # --- Both failed ---
        if raw_content is None:
            logger.error("LLMInterface: all providers failed — returning fallback answer")
            return self._fallback_answer("All LLM providers failed.")

        logger.debug("LLMInterface: raw response (provider=%s): %.200s", provider_used, raw_content)

        try:
            return self._parse_response(raw_content, retrieval_result)
        except Exception as exc:  # noqa: BLE001
            logger.error("LLMInterface: response parsing failed: %s", exc)
            return self._fallback_answer(f"Response parsing error: {exc}")

    # ------------------------------------------------------------------
    # Provider calls
    # ------------------------------------------------------------------

    def _call_openai(self, messages: List[dict]) -> str:
        """Call GPT-4o via the OpenAI Chat Completions API.

        Enables JSON mode (``response_format={"type": "json_object"}``) which
        guarantees the response is valid JSON when the prompt requests it.

        Parameters
        ----------
        messages:
            OpenAI-style messages list.

        Returns
        -------
        str
            Raw text content of the completion.

        Raises
        ------
        Exception
            Any OpenAI API or network error — caller handles the fallback.
        """
        client = self._ensure_openai_client()
        if client is None:
            raise RuntimeError("OpenAI client could not be initialised — key missing or package not installed")

        response = client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        return content

    def _call_ollama(self, messages: List[dict]) -> str:
        """Call a local model via Ollama's OpenAI-compatible endpoint.

        Ollama exposes ``/v1/chat/completions`` which the ``openai`` client
        can target by setting ``base_url``.  JSON mode is attempted but
        degraded gracefully if the model does not support it.

        Parameters
        ----------
        messages:
            OpenAI-style messages list.

        Returns
        -------
        str
            Raw text content of the completion.

        Raises
        ------
        Exception
            Any connection or model error — caller handles the fallback.
        """
        client = self._ensure_ollama_client()
        if client is None:
            raise RuntimeError("Ollama client could not be initialised — package missing or URL unreachable")

        # Try JSON mode first; some Ollama models support it
        try:
            response = client.chat.completions.create(
                model=self.ollama_model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        except Exception:
            # Fallback: no JSON mode (older Ollama / model does not support it)
            logger.debug("Ollama: JSON mode not supported — retrying without response_format")
            response = client.chat.completions.create(
                model=self.ollama_model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

        content = response.choices[0].message.content or ""
        return content

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        raw: str,
        retrieval_result: Optional[Any],
    ) -> MedicalAnswer:
        """Parse the LLM's raw text response into a MedicalAnswer.

        Handles:
        * Clean JSON objects
        * JSON wrapped in markdown code fences (```json ... ```)
        * JSON embedded in explanatory prose
        * Missing or malformed fields — fills with safe defaults

        Parameters
        ----------
        raw:
            Raw string returned by the LLM provider.
        retrieval_result:
            Optional RetrievalResult used for confidence and mode.

        Returns
        -------
        MedicalAnswer
        """
        json_str = self._extract_json(raw)

        try:
            data: Dict[str, Any] = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON decode failed: {exc}\nRaw snippet: {raw[:300]}") from exc

        # --- Citations ---
        raw_citations = data.get("citations", [])
        if not isinstance(raw_citations, list):
            raw_citations = []

        citations: List[Citation] = []
        for i, c in enumerate(raw_citations):
            if not isinstance(c, dict):
                continue
            try:
                citations.append(
                    Citation(
                        citation_id=str(c.get("citation_id", f"c{i + 1}")),
                        title=str(c.get("title", "Unknown source")),
                        url=c.get("url") or None,
                        source=str(c.get("source", "unknown")),
                        year=int(c["year"]) if c.get("year") else None,
                        authors=list(c.get("authors", [])),
                        relevance_score=float(c.get("relevance_score", 0.0)),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not parse citation %d: %s", i, exc)

        # --- Graph path ---
        raw_path = data.get("graph_path", [])
        graph_path: List[str] = [str(p) for p in raw_path] if isinstance(raw_path, list) else []

        # --- Answer text ---
        answer_text = str(data.get("answer", _FALLBACK_ANSWER_TEXT)).strip()
        if not answer_text:
            answer_text = _FALLBACK_ANSWER_TEXT

        # --- Confidence ---
        llm_confidence = float(data.get("confidence", 0.0))
        confidence = self._build_confidence(llm_confidence, retrieval_result)

        # --- Retrieval mode ---
        mode = RetrievalMode.HYBRID  # default
        if retrieval_result is not None:
            try:
                mode = retrieval_result.retrieval_mode
            except AttributeError:
                pass

        # --- Graph subgraph ---
        raw_subgraph = None
        if retrieval_result is not None:
            try:
                raw_subgraph = retrieval_result.graph_subgraph
            except AttributeError:
                pass

        return MedicalAnswer(
            answer=answer_text,
            citations=citations,
            graph_path=graph_path,
            confidence=confidence,
            retrieval_mode=mode,
            raw_graph_subgraph=raw_subgraph,
        )

    # ------------------------------------------------------------------
    # Helpers — client initialisation (lazy)
    # ------------------------------------------------------------------

    def _ensure_openai_client(self) -> Optional[Any]:
        """Lazily initialise and return the OpenAI client.

        Returns ``None`` if the key is missing or the package is not installed.
        """
        if self._openai_client is not None:
            return self._openai_client

        if not OPENAI_AVAILABLE:
            logger.warning("openai package not installed")
            return None

        if not self.openai_api_key:
            logger.warning("OPENAI_API_KEY not set — OpenAI disabled")
            return None

        try:
            self._openai_client = openai_lib.OpenAI(api_key=self.openai_api_key)
            logger.info("OpenAI client initialised (model=%s)", self.openai_model)
        except Exception as exc:  # noqa: BLE001
            logger.error("OpenAI client init failed: %s", exc)
            return None

        return self._openai_client

    def _ensure_ollama_client(self) -> Optional[Any]:
        """Lazily initialise and return the Ollama client.

        Uses the openai package pointed at the Ollama base URL.
        Returns ``None`` if the package is not installed.
        """
        if self._ollama_client is not None:
            return self._ollama_client

        if not OPENAI_AVAILABLE:
            logger.warning("openai package not installed — cannot create Ollama client")
            return None

        try:
            self._ollama_client = openai_lib.OpenAI(
                base_url=f"{self.ollama_base_url}/v1",
                api_key="ollama",  # Ollama does not require auth
            )
            logger.info(
                "Ollama client initialised (url=%s/v1, model=%s)",
                self.ollama_base_url,
                self.ollama_model,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Ollama client init failed: %s", exc)
            return None

        return self._ollama_client

    # ------------------------------------------------------------------
    # Helpers — prompt normalisation
    # ------------------------------------------------------------------

    def _to_messages(self, prompt: Union[str, List[dict]]) -> List[dict]:
        """Normalise *prompt* to an OpenAI messages list.

        If *prompt* is already a list of dicts it is returned unchanged.
        A plain string is wrapped in a single ``user`` message so the
        caller can use either ``PromptBuilder.build()`` or
        ``PromptBuilder.build_messages()`` interchangeably.

        Parameters
        ----------
        prompt:
            Flat string or OpenAI-style messages list.

        Returns
        -------
        List[dict]
            OpenAI ``messages`` parameter ready for chat completions.
        """
        if isinstance(prompt, list):
            return prompt

        # Flat string — wrap as user message
        return [{"role": "user", "content": prompt}]

    # ------------------------------------------------------------------
    # Helpers — JSON extraction
    # ------------------------------------------------------------------

    def _extract_json(self, text: str) -> str:
        """Strip markdown fences and extract the outermost JSON object.

        Handles the following common LLM output patterns:
        * Clean JSON: ``{"answer": ...}``
        * Fenced with language tag: ``\\`\\`\\`json\\n{...}\\n\\`\\`\\`\\``
        * Fenced without tag: ``\\`\\`\\`\\n{...}\\n\\`\\`\\`\\``
        * JSON preceded or followed by explanatory prose

        Parameters
        ----------
        text:
            Raw LLM output string.

        Returns
        -------
        str
            The extracted JSON string ready for ``json.loads()``.
        """
        # Remove markdown code fences
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"\s*```\s*$", "", text.strip(), flags=re.MULTILINE)
        text = text.strip()

        # Find the outermost JSON object using brace depth tracking
        if "{" not in text:
            return text  # let json.loads raise a meaningful error

        start = text.index("{")
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        # Braces didn't balance — return from start to end and let json.loads handle it
        return text[start:]

    # ------------------------------------------------------------------
    # Helpers — confidence calculation
    # ------------------------------------------------------------------

    def _build_confidence(
        self,
        llm_confidence: float,
        retrieval_result: Optional[Any],
    ) -> float:
        """Compute the final confidence score for the MedicalAnswer.

        Per the project specification, confidence is the **product of edge
        confidences along the retrieved graph path** (``path_confidence``).
        When a graph subgraph is available this value is authoritative.
        The LLM-reported confidence is used only when no graph path exists.

        Parameters
        ----------
        llm_confidence:
            The ``confidence`` value extracted from the LLM's JSON response.
        retrieval_result:
            Optional RetrievalResult from the orchestrator.

        Returns
        -------
        float
            Confidence clamped to [0.0, 1.0].
        """
        # Prefer graph path_confidence (authoritative per spec)
        if retrieval_result is not None:
            try:
                subgraph = retrieval_result.graph_subgraph
                if subgraph is not None and subgraph.path_confidence > 0.0:
                    return max(0.0, min(1.0, subgraph.path_confidence))
            except AttributeError:
                pass

        # Fall back to LLM-reported confidence
        return max(0.0, min(1.0, llm_confidence))

    # ------------------------------------------------------------------
    # Helpers — graceful fallback
    # ------------------------------------------------------------------

    def _fallback_answer(self, error_msg: str) -> MedicalAnswer:
        """Return a safe placeholder MedicalAnswer when all providers fail.

        The answer text explains the failure; citations and graph_path are
        empty; confidence is 0.0.  The disclaimer is always included.

        Parameters
        ----------
        error_msg:
            Short technical reason for the failure (logged but not shown
            to the end user in the answer text).

        Returns
        -------
        MedicalAnswer
        """
        logger.error("Returning fallback answer. Reason: %s", error_msg)
        return MedicalAnswer(
            answer=_FALLBACK_ANSWER_TEXT,
            citations=[],
            graph_path=[],
            confidence=0.0,
            retrieval_mode=RetrievalMode.HYBRID,
            raw_graph_subgraph=None,
        )