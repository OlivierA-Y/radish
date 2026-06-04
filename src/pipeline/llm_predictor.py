"""
LLM prediction module — zero-shot IDH mutation classification.
Supports OpenAI (GPT-4o, GPT-5) and OpenRouter (Claude, Gemini, Llama, …).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model aliases
# ---------------------------------------------------------------------------

OPENAI_MODEL_ALIASES: Dict[str, str] = {
    "gpt-4o":            "gpt-4o",
    "gpt-4o-mini":       "gpt-4o-mini",
    "gpt-5":             "gpt-5-chat-latest",
    "gpt-5-chat-latest": "gpt-5-chat-latest",
}

# Convenience aliases for popular OpenRouter models
OPENROUTER_MODEL_ALIASES: Dict[str, str] = {
    "claude-opus-4":        "anthropic/claude-opus-4",
    "claude-sonnet-4-5":    "anthropic/claude-sonnet-4-5",
    "claude-haiku-4-5":     "anthropic/claude-haiku-4-5-20251001",
    "gemini-pro":           "google/gemini-pro-1.5",
    "gemini-flash":         "google/gemini-flash-1.5",
    "llama-70b":            "meta-llama/llama-3.3-70b-instruct",
    "llama-8b":             "meta-llama/llama-3.1-8b-instruct",
    "mistral-large":        "mistralai/mistral-large",
    "qwen-72b":             "qwen/qwen-2.5-72b-instruct",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL       = "gpt-4o"
DEFAULT_PROVIDER    = "openai"


def _resolve_model(model: str, provider: str) -> str:
    """Expand short aliases to full model IDs."""
    if provider == "openrouter":
        return OPENROUTER_MODEL_ALIASES.get(model, model)
    return OPENAI_MODEL_ALIASES.get(model, model)


def _infer_provider(model: str, provider: Optional[str]) -> str:
    """
    If provider is not given explicitly, infer from model string:
    a slash in the model name (e.g. 'anthropic/claude-opus-4') signals OpenRouter.
    """
    if provider:
        return provider
    return "openrouter" if "/" in model else "openai"


@dataclass
class IDHPrediction:
    subject_id: str
    model: str
    idh_status: str          # "IDH-mutant" | "IDH-wildtype"
    confidence: str          # "high" | "medium" | "low"
    reasoning: str
    raw_response: str
    latency_s: float
    ground_truth: Optional[int] = None    # 1 = mutant, 0 = wildtype
    correct: Optional[bool] = field(default=None, init=False)

    def __post_init__(self):
        if self.ground_truth is not None:
            pred_label = 1 if "mutant" in self.idh_status.lower() else 0
            self.correct = pred_label == self.ground_truth

    @property
    def label(self) -> int:
        return 1 if "mutant" in self.idh_status.lower() else 0

    @property
    def confidence_score(self) -> float:
        return {"high": 0.9, "medium": 0.65, "low": 0.4}.get(self.confidence.lower(), 0.5)


def _parse_response(text: str) -> Dict[str, str]:
    """Extract JSON from LLM response (handles markdown code fences)."""
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip("`").strip()
    # Find first JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Fallback: regex field extraction
    idh = "IDH-wildtype"
    if re.search(r"idh.mutant", text, re.IGNORECASE):
        idh = "IDH-mutant"
    conf_match = re.search(r'"confidence"\s*:\s*"(\w+)"', text, re.IGNORECASE)
    confidence = conf_match.group(1) if conf_match else "low"
    reason_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    reasoning = reason_match.group(1) if reason_match else text[:300]
    return {"idh_status": idh, "confidence": confidence, "reasoning": reasoning}


def _load_env_key(var_name: str) -> Optional[str]:
    """Read an API key from environment or .env file in the working directory."""
    value = os.getenv(var_name)
    if value:
        return value
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith(var_name):
                    return line.split("=", 1)[-1].strip().strip('"').strip("'")
    return None


class LLMPredictor:
    """
    Zero-shot IDH prediction via a chat completion API.

    provider="openai"     — OpenAI API (GPT-4o, GPT-5, …)
                            Key: OPENAI_API_KEY
    provider="openrouter" — OpenRouter (Claude, Gemini, Llama, Mistral, …)
                            Key: OPENROUTER_API_KEY
                            Docs: https://openrouter.ai/docs

    Auto-detection: if model contains '/' (e.g. 'anthropic/claude-opus-4'),
    the provider defaults to 'openrouter' even if not specified.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        provider: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 3,
        retry_delay_s: float = 2.0,
        http_referer: str = "https://github.com/idh-llm-pipeline",
        app_title: str = "IDH-LLM-Pipeline",
    ):
        self.provider       = _infer_provider(model, provider)
        self.model          = _resolve_model(model, self.provider)
        self.temperature    = temperature
        self.max_tokens     = max_tokens
        self.max_retries    = max_retries
        self.retry_delay_s  = retry_delay_s
        self.http_referer   = http_referer
        self.app_title      = app_title
        self._client        = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai>=1.30.0 is required: pip install openai") from e

        if self.provider == "openrouter":
            api_key = _load_env_key("OPENROUTER_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENROUTER_API_KEY not found. "
                    "Set it in the environment or in a .env file.\n"
                    "Get a key at https://openrouter.ai/keys"
                )
            self._client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=api_key,
                default_headers={
                    "HTTP-Referer": self.http_referer,
                    "X-Title":      self.app_title,
                },
            )
            logger.info("Using OpenRouter — model: %s", self.model)
        else:
            api_key = _load_env_key("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY not found. "
                    "Set it in the environment or in a .env file."
                )
            self._client = OpenAI(api_key=api_key)
            logger.info("Using OpenAI — model: %s", self.model)

        return self._client

    def predict(
        self,
        subject_id: str,
        narrative: str,
        ground_truth: Optional[int] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ) -> IDHPrediction:
        from src.prompts.templates import build_prompt

        sys_p, usr_p = build_prompt(narrative)
        if system_prompt:
            sys_p = system_prompt
        if user_prompt_template:
            usr_p = user_prompt_template.format(narrative=narrative)

        messages = [
            {"role": "system",  "content": sys_p},
            {"role": "user",    "content": usr_p},
        ]

        client = self._get_client()
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.perf_counter()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                latency = time.perf_counter() - t0
                raw = response.choices[0].message.content or ""
                parsed = _parse_response(raw)
                return IDHPrediction(
                    subject_id=subject_id,
                    model=self.model,
                    idh_status=parsed.get("idh_status", "IDH-wildtype"),
                    confidence=parsed.get("confidence", "low"),
                    reasoning=parsed.get("reasoning", ""),
                    raw_response=raw,
                    latency_s=latency,
                    ground_truth=ground_truth,
                )
            except Exception as e:
                last_err = e
                logger.warning(
                    "LLM call attempt %d/%d failed: %s", attempt, self.max_retries, e
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_s * attempt)

        raise RuntimeError(f"LLM prediction failed after {self.max_retries} attempts: {last_err}")

    def predict_batch(
        self,
        subjects: List[Dict],
        delay_between_s: float = 0.5,
    ) -> List[IDHPrediction]:
        """
        subjects: list of dicts with keys: subject_id, narrative, ground_truth (optional)
        """
        results = []
        for i, subj in enumerate(subjects):
            logger.info(
                "Predicting subject %d/%d: %s",
                i + 1, len(subjects), subj["subject_id"],
            )
            pred = self.predict(
                subject_id=subj["subject_id"],
                narrative=subj["narrative"],
                ground_truth=subj.get("ground_truth"),
            )
            results.append(pred)
            if delay_between_s > 0 and i < len(subjects) - 1:
                time.sleep(delay_between_s)
        return results
