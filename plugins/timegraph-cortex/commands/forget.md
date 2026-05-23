---
description: Mark a stored fact as wrong (lowers confidence, unpins) using the timegraph attest tool
argument-hint: <fact-pattern-or-id>
---

The user wants to forget or correct a stored fact: "$ARGUMENTS".

Steps:
1. If "$ARGUMENTS" looks like a UUID (fact_id), skip to step 3.
2. Otherwise, call `mcp__timegraph__recall` with `query="$ARGUMENTS"` and `k=5` to locate candidate facts.
3. Show the candidates to the user with their fact_ids and ask which one(s) to forget. If only one candidate is highly relevant, proceed without asking.
4. For each fact_id the user (or you) selected, call `mcp__timegraph__attest` with `fact_id=...`, `confirmed=False`, `attestation="user-issued /forget"`.
5. Confirm to the user what was forgotten.
