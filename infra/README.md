# infra/

Runtime: **LM Studio** at `http://127.0.0.1:1234`.

Launch LM Studio from its GUI; load Qwopus3.6-27B-v1-preview Q5_K_M and (optionally)
Qwen3-7B-Instruct Q5_K_M. Models are addressed by their LM Studio identifier in the
`model` field of each request. Structured output is enforced via OpenAI-compat
`response_format: { type: "json_schema", json_schema: {...}, strict: true }` — see
`src/timegraph/llm/schemas.py`.

### If you ever want to switch back to llama-server

For higher throughput or scriptable spin-up, run `llama-server.exe` directly:

```powershell
llama-server.exe ^
  --model .\models\qwopus3.6-27b-Q5_K_M.gguf ^
  --port 1234 --alias qwopus3.6-27b-v1-preview ^
  --ctx-size 32768 --n-gpu-layers 999 --threads 8
```

Note: llama-server uses the raw `grammar` request field instead of `response_format`.
If switching, adapt `judge.py` / `extractor.py` accordingly (or set both — llama-server
accepts both for safety).

### `grammars/`

Empty in the LM Studio path. Reserved for future use if we switch back to llama-server
and want GBNF grammars co-located with the project.
