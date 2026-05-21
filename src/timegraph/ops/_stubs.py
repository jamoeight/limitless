"""Stub markers for the 9 B.4-v2 ops + 5 B.2-v2 ops.

Each op module is created as a placeholder during Phase 0 skeleton. Phase 1
implements the B.4-v2 ones; Phase 2 the B.2-v2 ones. Each stub raises
NotImplementedError so accidental calls during early dev fail loudly.

Op-to-phase mapping:
  Phase 1 — B.4-v2 (capability layer):
    add_episode, add_fact, graph_query, infer, fuse,
    invalidate, attest, delete, claim_release
  Phase 2 — B.2-v2 (safety layer):
    attest_quorum, subscribe_signals, accept_signal, dismiss_signal,
    context_window_update
"""

from __future__ import annotations

OP_PHASE: dict[str, int] = {
    # B.4-v2 (Phase 1)
    "add_episode": 1,
    "add_fact": 1,
    "graph_query": 1,
    "infer": 1,
    "fuse": 1,
    "invalidate": 1,
    "attest": 1,
    "delete": 1,
    "claim_release": 1,
    # B.2-v2 (Phase 2)
    "attest_quorum": 2,
    "subscribe_signals": 2,
    "accept_signal": 2,
    "dismiss_signal": 2,
    "context_window_update": 2,
}


class OpNotImplemented(NotImplementedError):
    """Raised by stub ops to make 'forgot to implement' failures loud."""

    def __init__(self, op_name: str):
        phase = OP_PHASE.get(op_name, "?")
        super().__init__(
            f"Op '{op_name}' is a Phase-{phase} stub; not yet implemented. "
            f"See ~/.claude/plans/plan-out-the-build-splendid-rivest.md for sequencing."
        )
