"""Parse GraphWalks rows: extract the graph edge list, the operation
(BFS or parents), and the ground-truth node set. No LLM, no DB — pure
string parsing + scoring utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_EDGES_HEADER_RE = re.compile(r"Here is the graph to operate on:\s*The graph has the following edges:\s*",
                              re.IGNORECASE)
_OP_HEADER_RE = re.compile(r"\nOperation:\s*\n", re.IGNORECASE)
_EDGE_LINE_RE = re.compile(r"^([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)\s*$")
_BFS_RE = re.compile(
    r"BFS\s+from\s+node\s+([A-Za-z0-9_]+).*?depth\s+(\d+)",
    re.IGNORECASE | re.DOTALL,
)
_PARENTS_RE = re.compile(
    r"(?:parents|Find the parents)\s+of\s+node\s+([A-Za-z0-9_]+)",
    re.IGNORECASE,
)

OpKind = Literal["bfs", "parents"]


@dataclass
class GwTask:
    """One GraphWalks task — parsed."""
    edges: list[tuple[str, str]]
    nodes: set[str]
    op: OpKind
    start_node: str
    depth: int | None  # for bfs
    answer: set[str]
    prompt_chars: int
    raw_prompt: str   # kept for in-context baseline runs


def parse_row(row: dict) -> GwTask:
    """Parse one HuggingFace dataset row into a GwTask."""
    prompt: str = row["prompt"]
    answer_nodes: list[str] = row["answer_nodes"]
    problem_type: str = row["problem_type"]

    # The prompt contains a 3-shot example block, then the real graph.
    # We slice from "Here is the graph to operate on:" onward to skip the examples.
    m = _EDGES_HEADER_RE.search(prompt)
    if not m:
        raise ValueError("could not find graph header in prompt")
    after = prompt[m.end():]

    # The operation section is delimited by a blank line then "Operation:"
    op_m = _OP_HEADER_RE.search(after)
    if not op_m:
        raise ValueError("could not find Operation header in prompt")
    edge_block = after[: op_m.start()]
    op_block = after[op_m.end():]

    edges: list[tuple[str, str]] = []
    nodes: set[str] = set()
    for line in edge_block.splitlines():
        m2 = _EDGE_LINE_RE.match(line)
        if not m2:
            continue
        a, b = m2.group(1), m2.group(2)
        edges.append((a, b))
        nodes.add(a)
        nodes.add(b)

    op: OpKind
    start: str
    depth: int | None
    if problem_type == "bfs":
        bm = _BFS_RE.search(op_block)
        if not bm:
            raise ValueError(f"could not parse BFS op from: {op_block[:200]!r}")
        op = "bfs"
        start = bm.group(1)
        depth = int(bm.group(2))
    elif problem_type == "parents":
        pm = _PARENTS_RE.search(op_block)
        if not pm:
            raise ValueError(f"could not parse parents op from: {op_block[:200]!r}")
        op = "parents"
        start = pm.group(1)
        depth = None
    else:
        raise ValueError(f"unknown problem_type: {problem_type}")

    return GwTask(
        edges=edges,
        nodes=nodes,
        op=op,
        start_node=start,
        depth=depth,
        answer=set(answer_nodes),
        prompt_chars=int(row["prompt_chars"]),
        raw_prompt=prompt,
    )


def f1(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    """Set-based precision / recall / F1. Both empty -> all 1.0 (vacuously correct)."""
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    if p + r == 0:
        return 0.0, 0.0, 0.0
    return p, r, 2 * p * r / (p + r)


def exact_match(pred: set[str], gold: set[str]) -> bool:
    return pred == gold
