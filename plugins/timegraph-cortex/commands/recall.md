---
description: Manually recall stored memories matching a query (uses the timegraph MCP recall tool)
argument-hint: <query>
---

Call the `mcp__timegraph__recall` tool with `query="$ARGUMENTS"` and `k=12`.

Display the results to the user as a markdown list. For each result, include:
- The fact triple (subject, predicate, object)
- The valid_at timestamp (date only)
- The source / session_id
- The confidence

If results are empty, tell the user that nothing matched and suggest:
1. Broadening the query
2. Checking `timegraph stats` to verify there's any memory at all
3. Verifying backends are up with `timegraph status`
