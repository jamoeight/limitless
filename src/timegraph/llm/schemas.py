"""JSON schemas for LM Studio's `response_format: {type: json_schema, strict: true}`.

Why a `thinking` field?
-----------------------
Qwopus3.6-27B is a reasoning model that natively wraps responses in
`<think>...</think>` blocks. Strict JSON output precludes raw thinking blocks
(every token must match the schema from position 0). To preserve the model's
reasoning behavior under strict-mode constraint, each schema includes a
leading optional `thinking` string field. The model writes its chain-of-thought
there; clients drop it before consuming the structured payload.

Strict-mode requirements LM Studio enforces:
  - additionalProperties: false
  - every property listed in `required`
  - no `$ref` to external schemas

We meet all three throughout.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Judge (stage-2 of B.4-v2 infer(mode="conflict_set"))
# ---------------------------------------------------------------------------

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["thinking", "resolution", "reason", "confidence"],
    "properties": {
        "thinking": {
            "type": "string",
            "description": (
                "Brief chain-of-thought (≤512 tokens recommended) reasoning about "
                "which conflict-pair candidate is best supported by the evidence. "
                "Consider temporal recency, source authority, and corroboration. "
                "If evidence is insufficient, say so and return 'unresolved'."
            ),
        },
        "resolution": {
            "type": "string",
            "enum": ["e1_correct", "e2_correct", "both_partial", "unresolved"],
            "description": "Winner of the conflict, or unresolved if evidence insufficient.",
        },
        "reason": {
            "type": "string",
            "description": "One concise sentence stating the basis for the resolution.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Subjective confidence in the resolution.",
        },
    },
}

JUDGE_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "ConflictResolution",
        "strict": True,
        "schema": JUDGE_SCHEMA,
    },
}


# ---------------------------------------------------------------------------
# Extractor (add_episode fact extraction)
# ---------------------------------------------------------------------------

EXTRACTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["thinking", "facts"],
    "properties": {
        "thinking": {
            "type": "string",
            "description": (
                "Brief notes on what's extractable from this episode. If using a "
                "non-thinking model (e.g., Qwen3-7B-Instruct), keep this to one line "
                "or empty string."
            ),
        },
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["subject", "predicate", "object", "confidence"],
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Canonical entity name (no pronouns; resolve coreferences).",
                    },
                    "predicate": {
                        "type": "string",
                        "description": "Short verb or attribute name (snake_case preferred), e.g., 'lives_in', 'works_at', 'has_email'.",
                    },
                    "object": {
                        "type": "string",
                        "description": "Object of the predicate; can be entity, value, or literal.",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.6,
                        "maximum": 1.0,
                        "description": "Subjective extraction confidence; omit facts <0.6.",
                    },
                },
            },
        },
    },
}

EXTRACTOR_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "ExtractedFacts",
        "strict": True,
        "schema": EXTRACTOR_SCHEMA,
    },
}
