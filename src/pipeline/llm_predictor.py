"""
LLM prediction module — zero-shot IDH mutation classification via OpenRouter.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model aliases
# ---------------------------------------------------------------------------

OPENROUTER_MODEL_ALIASES: Dict[str, str] = {
    "claude-opus-4":        "anthropic/claude-opus-4",
    "claude-sonnet-4-5":    "anthropic/claude-sonnet-4.5",
    "claude-haiku-4-5":     "anthropic/claude-haiku-4.5",
    "gemini-pro":           "google/gemini-pro-1.5",
    "gemini-flash":         "google/gemini-flash-1.5",
    "llama-70b":            "meta-llama/llama-3.3-70b-instruct",
    "llama-8b":             "meta-llama/llama-3.1-8b-instruct",
    "mistral-large":        "mistralai/mistral-large",
    "qwen-72b":             "qwen/qwen-2.5-72b-instruct",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL       = "anthropic/claude-opus-4"

IDH_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "idh_prediction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "idh_status": {
                    "type": "string",
                    "description": "IDH mutation status: 'IDH-mutant' or 'IDH-wildtype'",
                    "enum": ["IDH-mutant", "IDH-wildtype"],
                },
                "reasoning": {
                    "type": "string",
                    "description": "Compact clinical reasoning based on MRI imaging features",
                },
            },
            "required": ["idh_status", "reasoning"],
            "additionalProperties": False,
        },
    },
}


def _resolve_model(model: str) -> str:
    """Expand short aliases to full OpenRouter model IDs."""
    return OPENROUTER_MODEL_ALIASES.get(model, model)


@dataclass
class IDHPrediction:
    subject_id: str
    model: str
    idh_status: str          # "IDH-mutant" | "IDH-wildtype"
    reasoning: str
    raw_response: str
    latency_s: float
    ground_truth: Optional[int] = None    # 1 = mutant, 0 = wildtype
    model_version: Optional[str] = None
    correct: Optional[bool] = field(default=None, init=False)

    def __post_init__(self):
        if self.ground_truth is not None:
            pred_label = 1 if "mutant" in self.idh_status.lower() else 0
            self.correct = pred_label == self.ground_truth

    @property
    def label(self) -> int:
        return 1 if "mutant" in self.idh_status.lower() else 0




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
    Zero-shot IDH prediction via OpenRouter (Claude, Gemini, Llama, Mistral, …).
    Key: OPENROUTER_API_KEY — https://openrouter.ai/keys
    seed=<int> pins the random seed for determinism.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_delay_s: float = 2.0,
        http_referer: str = "https://github.com/idh-llm-pipeline",
        app_title: str = "IDH-LLM-Pipeline",
        log_path: Optional[str | Path] = None,
        seed: Optional[int] = 42,
    ):
        self.model          = _resolve_model(model)
        self.temperature    = temperature
        self.max_tokens     = max_tokens
        self.max_retries    = max_retries
        self.retry_delay_s  = retry_delay_s
        self.http_referer   = http_referer
        self.app_title      = app_title
        self.log_path       = Path(log_path) if log_path else None
        self.seed           = seed
        self._client        = None
        self._log_lock      = threading.Lock()
        self._init_lock     = threading.Lock()

    def _get_client(self):
        if self._client is not None:
            return self._client
        with self._init_lock:
            if self._client is not None:
                return self._client
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("openai>=1.30.0 is required: pip install openai") from e
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
            logger.info("OpenRouter — model: %s", self.model)
        return self._client

    def _append_log(self, entry: Dict) -> None:
        if self.log_path is None:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_lock:
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to write log entry for %s: %s", entry.get("subject_id"), exc)

    def raw_query(self, messages: List[Dict]) -> str:
        """Make a direct API call and return the raw response string."""
        client = self._get_client()
        kwargs: Dict = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if self.seed is not None:
            kwargs["seed"] = self.seed
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def predict(
        self,
        subject_id: str,
        ground_truth: Optional[int] = None,
        features_dict: Optional[Dict] = None,
        clinical_dict: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ) -> IDHPrediction:
        from src.prompts.templates import build_prompt

        sys_p, usr_p = build_prompt(features_dict or {}, clinical_dict)
        if system_prompt:
            sys_p = system_prompt
        if user_prompt_template:
            payload = dict(features_dict or {})
            if clinical_dict:
                payload = {"clinical_dict": clinical_dict, **payload}
            usr_p = user_prompt_template + f"\n\nHere is the json data:\n{json.dumps(payload, indent=4)}"

        messages = [
            {"role": "system",  "content": sys_p},
            {"role": "user",    "content": usr_p},
        ]

        client = self._get_client()
        last_err = None
        prediction = None
        log_entry: Dict = {}

        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.perf_counter()
                kwargs: Dict = dict(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format=IDH_RESPONSE_FORMAT,
                )
                if self.seed is not None:
                    kwargs["seed"] = self.seed
                response = client.chat.completions.create(**kwargs)
                latency = time.perf_counter() - t0
                raw = response.choices[0].message.content or ""
                parsed = json.loads(raw)
                model_version = getattr(response, "model", self.model)

                prediction = IDHPrediction(
                    subject_id=subject_id,
                    model=self.model,
                    idh_status=parsed.get("idh_status", "IDH-wildtype"),
                    reasoning=parsed.get("reasoning", ""),
                    raw_response=raw,
                    latency_s=latency,
                    ground_truth=ground_truth,
                    model_version=model_version,
                )
                log_entry = {
                    "timestamp":     datetime.now(timezone.utc).isoformat(),
                    "subject_id":    subject_id,
                    "model":         self.model,
                    "model_version": model_version,
                    "seed":          self.seed,
                    "system_prompt": sys_p,
                    "user_prompt":   usr_p,
                    "raw_response":  raw,
                    "parsed":        parsed,
                    "latency_s":     latency,
                }
                break   # success — exit retry loop
            except Exception as e:
                last_err = e
                logger.warning(
                    "LLM call attempt %d/%d failed: %s", attempt, self.max_retries, e
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_s * attempt)

        if prediction is None:
            raise RuntimeError(f"LLM prediction failed after {self.max_retries} attempts: {last_err}")

        # Write to log outside the retry loop so a log I/O error cannot trigger a retry
        self._append_log(log_entry)
        return prediction

    def predict_batch(
        self,
        subjects: List[Dict],
        max_workers: int = 1,
        delay_between_s: float = 0.5,
    ) -> List[IDHPrediction]:
        """
        subjects: list of dicts with keys: subject_id, features_dict, ground_truth (optional),
                  clinical_dict (optional)

        max_workers > 1 fires all LLM calls concurrently via ThreadPoolExecutor.
        """
        if max_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            results: List[IDHPrediction | None] = [None] * len(subjects)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.predict,
                        subject_id=subj["subject_id"],
                        ground_truth=subj.get("ground_truth"),
                        features_dict=subj.get("features_dict"),
                        clinical_dict=subj.get("clinical_dict"),
                    ): i
                    for i, subj in enumerate(subjects)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
            return [r for r in results if r is not None]

        results_seq = []
        for i, subj in enumerate(subjects):
            logger.info(
                "Predicting subject %d/%d: %s",
                i + 1, len(subjects), subj["subject_id"],
            )
            pred = self.predict(
                subject_id=subj["subject_id"],
                ground_truth=subj.get("ground_truth"),
                features_dict=subj.get("features_dict"),
                clinical_dict=subj.get("clinical_dict"),
            )
            results_seq.append(pred)
            if delay_between_s > 0 and i < len(subjects) - 1:
                time.sleep(delay_between_s)
        return results_seq
