"""
Tests for OpenRouter provider support in the LLM predictor.
No API key required.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.llm_predictor import (
    _infer_provider,
    _resolve_model,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL_ALIASES,
    OPENAI_MODEL_ALIASES,
    LLMPredictor,
)


# ── Provider auto-detection ───────────────────────────────────────────────────

def test_infer_openai_from_gpt_model():
    assert _infer_provider("gpt-4o", None) == "openai"

def test_infer_openai_explicit():
    assert _infer_provider("gpt-4o", "openai") == "openai"

def test_infer_openrouter_from_slash_model():
    assert _infer_provider("anthropic/claude-opus-4", None) == "openrouter"

def test_infer_openrouter_explicit_overrides():
    # explicit provider always wins over auto-detection
    assert _infer_provider("gpt-4o", "openrouter") == "openrouter"

def test_infer_openrouter_for_all_slash_models():
    slash_models = [
        "anthropic/claude-opus-4",
        "google/gemini-pro-1.5",
        "meta-llama/llama-3.3-70b-instruct",
        "mistralai/mistral-large",
        "qwen/qwen-2.5-72b-instruct",
    ]
    for m in slash_models:
        assert _infer_provider(m, None) == "openrouter", f"Expected openrouter for {m}"


# ── Model alias resolution ────────────────────────────────────────────────────

def test_openai_alias_gpt5():
    assert _resolve_model("gpt-5", "openai") == "gpt-5-chat-latest"

def test_openai_alias_passthrough():
    assert _resolve_model("gpt-4o", "openai") == "gpt-4o"

def test_openrouter_alias_claude_opus():
    assert _resolve_model("claude-opus-4", "openrouter") == "anthropic/claude-opus-4"

def test_openrouter_alias_gemini_flash():
    assert _resolve_model("gemini-flash", "openrouter") == "google/gemini-flash-1.5"

def test_openrouter_alias_llama():
    assert _resolve_model("llama-70b", "openrouter") == "meta-llama/llama-3.3-70b-instruct"

def test_openrouter_full_id_passthrough():
    full = "anthropic/claude-opus-4"
    assert _resolve_model(full, "openrouter") == full

def test_all_openai_aliases_resolve():
    for alias, expected in OPENAI_MODEL_ALIASES.items():
        assert _resolve_model(alias, "openai") == expected

def test_all_openrouter_aliases_resolve():
    for alias, expected in OPENROUTER_MODEL_ALIASES.items():
        assert _resolve_model(alias, "openrouter") == expected


# ── LLMPredictor construction ─────────────────────────────────────────────────

def test_predictor_default_is_openai():
    p = LLMPredictor(model="gpt-4o")
    assert p.provider == "openai"
    assert p.model    == "gpt-4o"

def test_predictor_auto_detects_openrouter():
    p = LLMPredictor(model="anthropic/claude-opus-4")
    assert p.provider == "openrouter"
    assert p.model    == "anthropic/claude-opus-4"

def test_predictor_explicit_openrouter_with_alias():
    p = LLMPredictor(model="claude-opus-4", provider="openrouter")
    assert p.provider == "openrouter"
    assert p.model    == "anthropic/claude-opus-4"

def test_predictor_explicit_openai_with_alias():
    p = LLMPredictor(model="gpt-5", provider="openai")
    assert p.provider == "openai"
    assert p.model    == "gpt-5-chat-latest"

def test_predictor_openrouter_default_headers():
    p = LLMPredictor(model="anthropic/claude-opus-4",
                     http_referer="https://myapp.example",
                     app_title="MyApp")
    assert p.http_referer == "https://myapp.example"
    assert p.app_title    == "MyApp"


# ── Client raises on missing key (no network needed) ─────────────────────────

def test_openai_client_raises_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = LLMPredictor(model="gpt-4o", provider="openai")
    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        p._get_client()

def test_openrouter_client_raises_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)   # no .env file here
    p = LLMPredictor(model="anthropic/claude-opus-4", provider="openrouter")
    with pytest.raises(EnvironmentError, match="OPENROUTER_API_KEY"):
        p._get_client()

def test_openrouter_base_url_format():
    assert OPENROUTER_BASE_URL.startswith("https://openrouter.ai")


# ── .env file loading ─────────────────────────────────────────────────────────

def test_load_env_key_from_file(tmp_path, monkeypatch):
    from src.pipeline.llm_predictor import _load_env_key
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text('OPENROUTER_API_KEY="or-test-key-123"\n')
    assert _load_env_key("OPENROUTER_API_KEY") == "or-test-key-123"

def test_load_env_key_prefers_env_var(tmp_path, monkeypatch):
    from src.pipeline.llm_predictor import _load_env_key
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-var-key")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=file-key\n")
    assert _load_env_key("OPENROUTER_API_KEY") == "env-var-key"

def test_load_env_key_returns_none_when_absent(tmp_path, monkeypatch):
    from src.pipeline.llm_predictor import _load_env_key
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    assert _load_env_key("OPENROUTER_API_KEY") is None


# ── CLI integration — dry_run still works with openrouter flag ────────────────

def test_main_dry_run_with_openrouter_flag(tmp_path):
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable, "main.py",
            "--mode", "synthetic",
            "--n_mutant", "2",
            "--n_wildtype", "2",
            "--provider", "openrouter",
            "--model", "anthropic/claude-opus-4",
            "--dry_run",
            "--output_dir", str(tmp_path),
        ],
        cwd=str(Path(__file__).parent.parent),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr[-1000:]
    run_dirs = list((tmp_path / "runs").glob("*/"))
    assert run_dirs, "runs/ subdirectory should be created"
    run_dir = run_dirs[0]
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "metrics.json").exists()
    lines = (run_dir / "predictions.csv").read_text().splitlines()
    assert len(lines) == 5, f"Expected header + 4 rows, got {len(lines)}"
