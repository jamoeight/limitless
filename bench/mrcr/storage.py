"""In-memory storage + retrieval for MRCR conversations.

The architectural point of MRCR is that the LLM should never see the
conversation. We index the conversation locally and answer the query by
deterministic lookup once an LLM call has parsed the question.

Per-row pipeline:
  1) Ingest the conversation (excluding the trailing query turn).
  2) Group consecutive (user, assistant) pairs and key them by the
     verbatim user-turn content.
  3) Given a parsed query {needle_request, position}, look up the matching
     user-turn key, sort by pair-index, return the assistant text at
     position - 1.

No LLM is used at storage or retrieval. The only LLM call is the query
parser (see query_parser.py).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnPair:
    pair_index: int     # 0-based order of (user, assistant) pairs in the conv
    user_text: str
    assistant_text: str
    user_msg_idx: int   # index in the original `messages` list
    asst_msg_idx: int


@dataclass
class MrcrIndex:
    """Per-row index built once at ingestion."""
    pairs: list[TurnPair]
    by_user_text: dict[str, list[TurnPair]]   # exact-match groups

    def lookup(self, needle_request: str, position: int) -> str | None:
        """Return the assistant text matching the Nth (1-indexed) user-turn
        whose content equals `needle_request`. Returns None if not found."""
        cand = self.by_user_text.get(needle_request)
        if not cand:
            # Fallback: try a normalized lookup (strip whitespace + lowercase).
            norm = _normalize(needle_request)
            for k, v in self.by_user_text.items():
                if _normalize(k) == norm:
                    cand = v
                    break
        if not cand:
            return None
        idx = position - 1
        if idx < 0 or idx >= len(cand):
            return None
        return cand[idx].assistant_text

    def candidate_count(self, needle_request: str) -> int:
        c = self.by_user_text.get(needle_request)
        if c is not None:
            return len(c)
        norm = _normalize(needle_request)
        for k, v in self.by_user_text.items():
            if _normalize(k) == norm:
                return len(v)
        return 0


def _normalize(s: str) -> str:
    s = " ".join(s.lower().split())
    # MRCR's user turns use "write a <thing>" verbatim even when the next word
    # is vowel-initial ("write a email", "write a article"). LLMs reflexively
    # fix it to "an"; collapse the article so exact-match still hits.
    if s.startswith("write an "):
        s = "write a " + s[len("write an "):]
    return s


def build_index(messages: list[dict[str, str]]) -> MrcrIndex:
    """Build a per-row needle lookup from the full conversation. We exclude
    the trailing user turn (the query itself)."""
    # Drop the final query turn; everything else is the "conversation" we
    # are answering against.
    body = messages[:-1] if messages and messages[-1]["role"] == "user" else list(messages)

    pairs: list[TurnPair] = []
    i = 0
    pair_index = 0
    while i < len(body) - 1:
        a, b = body[i], body[i + 1]
        if a["role"] == "user" and b["role"] == "assistant":
            pairs.append(TurnPair(
                pair_index=pair_index,
                user_text=a["content"],
                assistant_text=b["content"],
                user_msg_idx=i,
                asst_msg_idx=i + 1,
            ))
            pair_index += 1
            i += 2
        else:
            i += 1

    by_user: dict[str, list[TurnPair]] = {}
    for p in pairs:
        by_user.setdefault(p.user_text, []).append(p)

    return MrcrIndex(pairs=pairs, by_user_text=by_user)
