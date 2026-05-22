"""Parse MRCR rows: extract conversation, the trailing user query, and build
a per-row needle lookup table.

The conversation is a JSON list of `{role, content}` messages. By dataset
design, "needles" are identical user turns (e.g., 'write a poem about tapirs'
repeated 2/4/8 times throughout the conversation, each followed by a distinct
assistant generation). The LAST user turn is always the query of the form:

    Prepend <random_string> to the Nth (1 indexed) <type_phrase> about <topic>.
    Do not include any other text in your response.

The expected answer is `<random_string>` + verbatim text of the Nth needle's
assistant response.

This file also implements the official scoring rubric (SequenceMatcher.ratio
with the random-string prefix check).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


_QUERY_RE = re.compile(
    r"Prepend\s+(?P<prepend>\S+)\s+to\s+the\s+(?P<pos>\d+)(?:st|nd|rd|th)?\s*"
    r"(?:\(\s*1\s*indexed\s*\))?\s*(?P<phrase>.+?)(?=\.\s*Do\s+not)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class MrcrTask:
    """One MRCR task — parsed."""
    messages: list[dict[str, str]]      # full conversation including final query
    query_text: str                     # the LAST user message
    prepend: str                        # the random_string_to_prepend value
    position: int                       # 1-indexed
    phrase: str                         # e.g. "short scene in a play about blueberries"
    gold_answer: str                    # ground truth (random_string + verbatim needle)
    random_string: str                  # alias for prepend, matches the dataset field
    n_chars: int
    n_needles: int
    desired_msg_index: int
    total_messages: int


def parse_row(row: Any) -> MrcrTask:
    """Parse one MRCR row (dict or pandas Series). Falls back to the dataset
    fields directly when the regex on the query can't pull `prepend` / `pos`
    (which can happen on unusual phrasing) — `random_string_to_prepend` and
    `desired_msg_index` are authoritative in the dataset."""
    messages = json.loads(row["prompt"])
    query_text = messages[-1]["content"]
    random_string = row["random_string_to_prepend"]
    position = int(row["desired_msg_index"])

    # Best-effort regex pull for `phrase`; if regex fails we still have everything
    # we need from the dataset fields.
    phrase = ""
    m = _QUERY_RE.search(query_text)
    if m:
        phrase = m.group("phrase").strip()

    return MrcrTask(
        messages=messages,
        query_text=query_text,
        prepend=random_string,
        position=position,
        phrase=phrase,
        gold_answer=str(row["answer"]),
        random_string=random_string,
        n_chars=int(row["n_chars"]),
        n_needles=int(row["n_needles"]),
        desired_msg_index=position,
        total_messages=int(row["total_messages"]),
    )


def score(response: str, gold_answer: str, random_string: str) -> float:
    """Official MRCR grading. Hard 0 if response doesn't lead with the
    random-string token, else SequenceMatcher ratio against the gold."""
    if not response.startswith(random_string):
        return 0.0
    r = response.removeprefix(random_string)
    g = gold_answer.removeprefix(random_string)
    return float(SequenceMatcher(None, r, g).ratio())
