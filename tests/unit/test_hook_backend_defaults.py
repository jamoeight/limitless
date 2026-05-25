"""Unit tests for `timegraph.hooks.backend_defaults`.

This is the small helper every plugin hook runs before importing
timegraph.ops. It picks `anthropic_api` when OAuth/API-key creds are
available and `lm_studio` otherwise — the latter only matters in dev where
LM Studio is already loaded.
"""

from __future__ import annotations

import json

import pytest

from timegraph.hooks.backend_defaults import apply_hook_backend_defaults


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip the env vars the helper touches so each test starts clean."""
    for k in (
        "TG_JUDGE_BACKEND",
        "TG_EXTRACTOR_BACKEND",
        "TG_JUDGE_ANTHROPIC_MODEL",
        "TG_EXTRACTOR_ANTHROPIC_MODEL",
        "TG_JUDGE_CLAUDE_MODEL",
        "TG_EXTRACTOR_CLAUDE_MODEL",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)


def test_picks_anthropic_api_when_env_key_present(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-x")
    monkeypatch.setenv("TG_ANTHROPIC_CREDENTIALS_PATH", str(tmp_path / "absent.json"))

    import os

    apply_hook_backend_defaults()
    assert os.environ["TG_JUDGE_BACKEND"] == "anthropic_api"
    assert os.environ["TG_EXTRACTOR_BACKEND"] == "anthropic_api"
    assert os.environ["TG_JUDGE_ANTHROPIC_MODEL"] == "haiku"
    assert os.environ["TG_EXTRACTOR_ANTHROPIC_MODEL"] == "haiku"


def test_picks_anthropic_api_when_oauth_credentials_file_present(monkeypatch, tmp_path) -> None:
    creds = tmp_path / ".credentials.json"
    creds.write_text(
        json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-x"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TG_ANTHROPIC_CREDENTIALS_PATH", str(creds))

    import os

    apply_hook_backend_defaults()
    assert os.environ["TG_JUDGE_BACKEND"] == "anthropic_api"
    assert os.environ["TG_EXTRACTOR_BACKEND"] == "anthropic_api"


def test_falls_back_to_lm_studio_when_no_creds(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TG_ANTHROPIC_CREDENTIALS_PATH", str(tmp_path / "absent.json"))

    import os

    apply_hook_backend_defaults()
    assert os.environ["TG_JUDGE_BACKEND"] == "lm_studio"
    assert os.environ["TG_EXTRACTOR_BACKEND"] == "lm_studio"


def test_operator_override_always_wins(monkeypatch, tmp_path) -> None:
    """If TG_JUDGE_BACKEND was set before the hook fired, the helper must NOT
    overwrite it — operators have an escape hatch."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-x")
    monkeypatch.setenv("TG_JUDGE_BACKEND", "lm_studio")
    monkeypatch.setenv("TG_EXTRACTOR_BACKEND", "claude_cli")

    import os

    apply_hook_backend_defaults()
    assert os.environ["TG_JUDGE_BACKEND"] == "lm_studio"
    assert os.environ["TG_EXTRACTOR_BACKEND"] == "claude_cli"
