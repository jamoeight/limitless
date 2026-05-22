"""TimeGraph Cortex — infinite-context proxy for any frontier model.

A drop-in OpenAI/Anthropic-compatible HTTP proxy that wraps the timegraph
capability layer. Auto-ingests every message, auto-retrieves relevant context
per turn, and presents the result to any frontier model so existing clients
inherit infinite context with zero code changes.

See `plans/the-idea-of-sorted-eich.md` for the architecture.
"""

__version__ = "0.1.0-dev"
