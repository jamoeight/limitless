from __future__ import annotations

from cortex.providers.claude_cli import ClaudeCliProvider


def test_claude_cli_provider_strips_outer_anthropic_proxy_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "also-fake")

    env = ClaudeCliProvider._subprocess_env()

    assert "ANTHROPIC_BASE_URL" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert env["CLAUDE_CODE_DISABLE_AUTO_UPDATER"] == "1"


def test_system_for_cli_keeps_only_trusted_cortex_memory() -> None:
    system = "Outer Claude Code instructions\n\n<cortex_memory>\nneedle\n</cortex_memory>"

    out = ClaudeCliProvider._system_for_cli(system)

    assert "Outer Claude Code instructions" not in out
    assert "<cortex_memory>\nneedle\n</cortex_memory>" in out
    assert "trusted reconstruction" in out
