from __future__ import annotations

from cortex.canonical import CortexMessage, TextBlock
from cortex.recall import _literal_group_indices, _literal_needles


def _msg(text: str) -> CortexMessage:
    return CortexMessage(role="user", content=[TextBlock(text=text)])


def test_literal_needles_extract_payload_anchor() -> None:
    assert _literal_needles("quote PAYLOAD_ab12cd34 from earlier") == ["payload_ab12cd34"]


def test_literal_group_indices_prioritize_exact_anchor() -> None:
    groups = [
        [_msg("routine archive filler without the marker")],
        [_msg("field report begins at PAYLOAD_ab12cd34 and contains exact text")],
        [_msg("another routine archive filler")],
    ]

    assert _literal_group_indices("please quote PAYLOAD_ab12cd34", groups) == [1]
