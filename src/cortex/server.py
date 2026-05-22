"""TimeGraph Cortex proxy — FastAPI app.

Endpoints:
  POST /v1/messages    Anthropic Messages API (streaming + non-streaming)
  GET  /health         liveness
  GET  /v1/models      stub for client discovery (MVP-1: returns a fixed list)

MVP-1 scope: pure passthrough via the canonical layer. Inbound Anthropic
format → canonical → AnthropicProvider → canonical chunks → Anthropic SSE
back to the client.

The canonical detour is intentional: it exercises the translator round-trip
so MVP-3 virtualization and MVP-4 OpenAI routing slot in with zero churn.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from cortex.canonical import ChunkError, ChunkMessageStop, CortexChunk, CortexRequest
from cortex.config import CortexSettings, get_cortex_settings
from cortex.ingest import (
    Session,
    SessionRegistry,
    assistant_message_from_chunks,
    derive_group_id,
    ingest_request_messages,
)
from cortex.providers.anthropic import AnthropicProvider
from cortex.providers.base import Provider
from cortex.providers.openai import OpenAIProvider
from cortex.recall import real_recall
from cortex.translate.anthropic import (
    chunk_to_anthropic_sse,
    from_anthropic_request,
    response_from_chunks,
    to_anthropic_request,
)
from cortex.translate.openai import (
    chunk_to_openai_sse,
    from_openai_request,
    new_openai_egress_state,
    openai_response_from_chunks,
    to_openai_request,
)
from cortex.virtualize import RecallFn, VirtualizationReport, virtualize

log = structlog.get_logger(__name__)


# ---------- Provider registry ----------


class ProviderRegistry:
    """Holds long-lived provider clients keyed by name."""

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}

    def register(self, provider: Provider) -> None:
        self._providers[provider.name] = provider

    def get(self, name: str) -> Provider:
        if name not in self._providers:
            raise HTTPException(status_code=404, detail=f"unknown provider: {name}")
        return self._providers[name]

    def route_for_model(self, model: str, default: str = "anthropic") -> Provider:
        """Pick a provider based on model-name prefix.

        `default` is the fallback when no prefix matches — set via
        `CortexSettings.default_provider`. Useful when targeting an
        OpenAI-compatible local server (LM Studio, vLLM, llama.cpp) whose
        model names don't match the gpt-* / claude-* heuristics.
        """
        m = model.lower()
        if m.startswith("claude") or "anthropic" in m:
            return self.get("anthropic")
        if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
            return self.get("openai")
        return self.get(default)

    async def aclose(self) -> None:
        for p in self._providers.values():
            try:
                await p.aclose()
            except Exception as e:  # noqa: BLE001
                log.warning("provider close failed", provider=p.name, error=str(e))


# ---------- App state ----------


def _build_app(
    settings: CortexSettings | None = None,
    registry: ProviderRegistry | None = None,
    session_registry: SessionRegistry | None = None,
    recall_fn: RecallFn | None = None,
) -> FastAPI:
    """Construct the FastAPI app.

    Args:
        settings: override the loaded `CortexSettings`. Useful for tests.
        registry: pre-built provider registry. Replaces the default
            Anthropic-only registry. Tests inject a fake provider here.
        session_registry: pre-built session registry. Tests inject one with a
            stub `ingest_fn` so they don't need Neo4j/Qdrant/LM Studio.
        recall_fn: virtualization recall function. Defaults to `real_recall`
            which calls into the timegraph in-process. Tests inject a stub.
    """
    s = settings or get_cortex_settings()
    explicit_registry = registry
    explicit_session_registry = session_registry
    explicit_recall_fn = recall_fn

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if explicit_registry is not None:
            r = explicit_registry
        else:
            r = ProviderRegistry()
            r.register(AnthropicProvider(s))
            r.register(OpenAIProvider(s))
        sr = explicit_session_registry or SessionRegistry(s)
        app.state.registry = r
        app.state.session_registry = sr
        app.state.settings = s
        app.state.recall_fn = explicit_recall_fn or real_recall
        log.info("cortex.boot", host=s.host, port=s.port, providers=list(r._providers))
        try:
            yield
        finally:
            await r.aclose()
            log.info("cortex.shutdown")

    app = FastAPI(
        title="TimeGraph Cortex",
        version="0.1.0-dev",
        description="Infinite-context proxy for any frontier model",
        lifespan=lifespan,
    )
    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict[str, Any]:
        """Proxy /v1/models to the default upstream so OpenAI-format clients
        (and discovery plugins) can enumerate real model IDs.

        Falls back to a stub list if the upstream is unreachable.
        """
        s: CortexSettings = request.app.state.settings
        # For openai default, forward to the OpenAI base URL (which is what
        # LM Studio / vLLM / llama.cpp serve too).
        if s.default_provider == "openai":
            base = s.openai_base_url.rstrip("/")
            api_key = _try_extract_api_key(request) or "local"
            try:
                import httpx as _httpx

                async with _httpx.AsyncClient(timeout=5.0) as c:
                    r = await c.get(
                        f"{base}/v1/models",
                        headers={"authorization": f"Bearer {api_key}"},
                    )
                    if r.status_code == 200:
                        return r.json()
            except Exception:  # noqa: BLE001
                pass
        # Anthropic doesn't expose /v1/models in the same shape; return a
        # static hint covering common Claude 4.x ids so model-discovery
        # plugins have something to chew on.
        return {
            "object": "list",
            "data": [
                {"id": "claude-opus-4-7", "object": "model", "owned_by": "anthropic"},
                {"id": "claude-sonnet-4-6", "object": "model", "owned_by": "anthropic"},
                {"id": "claude-haiku-4-5", "object": "model", "owned_by": "anthropic"},
            ],
        }

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Any:
        return await _handle_messages(request, ingress="anthropic")

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(request: Request) -> Any:
        return await _handle_messages(request, ingress="openai")


async def _handle_messages(request: Request, *, ingress: str) -> Any:
    """Common request handler. `ingress` is "anthropic" or "openai" and
    controls which translator is used on parse + egress."""
    body_bytes = await request.body()
    try:
        raw = json.loads(body_bytes)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"invalid json body: {e}") from e
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")

    api_key = _extract_api_key(request)
    extra_headers = _passthrough_headers(request)

    try:
        if ingress == "anthropic":
            cortex_req = from_anthropic_request(raw)
        elif ingress == "openai":
            cortex_req = from_openai_request(raw)
        else:
            raise ValueError(f"unknown ingress format: {ingress}")
    except (KeyError, ValueError, TypeError) as e:
        log.warning("request parse failed", ingress=ingress, error=str(e))
        raise HTTPException(status_code=400, detail=f"invalid {ingress} request: {e}") from e

    # Cortex-specific headers (used heavily in later MVPs; collected here so
    # virtualization/ingest can read them without re-parsing the request).
    cortex_req.cortex_group_id = request.headers.get("x-cortex-group-id")
    cortex_req.cortex_session_id = request.headers.get("x-cortex-session-id")
    cortex_req.cortex_time_anchor_iso = request.headers.get("x-cortex-time-anchor")
    cortex_req.cortex_disable_virtualize = _bool_header(
        request.headers.get("x-cortex-disable-virtualize")
    )
    cortex_req.cortex_disable_ingest = _bool_header(
        request.headers.get("x-cortex-disable-ingest")
    )

    registry: ProviderRegistry = request.app.state.registry
    session_registry: SessionRegistry = request.app.state.session_registry
    settings: CortexSettings = request.app.state.settings
    recall_fn: RecallFn = request.app.state.recall_fn
    provider = registry.route_for_model(cortex_req.model, default=settings.default_provider)

    # Auto-ingest inbound messages (fire-and-forget). The session is keyed by
    # the cortex-provided (group_id, session_id) tuple if present; otherwise
    # we derive a stable group_id from the API key hash and use a fixed
    # session label so memory still persists across reconnects from the same
    # client.
    session: Session | None = None
    if settings.enable_auto_ingest and not cortex_req.cortex_disable_ingest:
        group_id = cortex_req.cortex_group_id or derive_group_id(api_key)
        session_id = cortex_req.cortex_session_id or "default"
        session = session_registry.get_or_create(group_id, session_id)
        ingest_request_messages(session, cortex_req.messages)

    # Virtualize cold history into a recap block before forwarding. The recap
    # is computed AFTER scheduling ingest — by the time we call recall_fn,
    # the in-flight task may or may not have completed; we read whatever is
    # in the graph as of now and proceed. The bounded-1-call retrieval keeps
    # this fast regardless of graph size.
    upstream_req = cortex_req
    virt_report: VirtualizationReport | None = None
    if settings.enable_virtualization and not cortex_req.cortex_disable_virtualize:
        tools_serialized = to_anthropic_request(cortex_req).get("tools", []) if cortex_req.tools else []
        upstream_req, virt_report = await virtualize(
            cortex_req,
            settings,
            recall_fn=recall_fn,
            context_limit=settings.upstream_context_limit,
            tools_serialized=tools_serialized,
        )

    client_wants_stream = cortex_req.stream
    if client_wants_stream:
        return _stream_response(
            provider, upstream_req, api_key, extra_headers, session, virt_report, ingress
        )
    return await _aggregate_response(
        provider, upstream_req, api_key, extra_headers, session, virt_report, ingress
    )


def _stream_response(
    provider: Provider,
    req: CortexRequest,
    api_key: str,
    extra_headers: dict[str, str],
    session: Session | None,
    virt_report: VirtualizationReport | None,
    ingress: str,
) -> EventSourceResponse:
    async def event_generator():
        collected: list[CortexChunk] = []
        egress_state = (
            new_openai_egress_state(req.model) if ingress == "openai" else None
        )
        try:
            async for chunk in provider.stream(req, api_key=api_key, extra_headers=extra_headers):
                if session is not None:
                    collected.append(chunk)

                if ingress == "anthropic":
                    sse = chunk_to_anthropic_sse(chunk)
                    if sse is None:
                        continue
                    event_name, payload = sse
                    yield {"event": event_name, "data": json.dumps(payload, ensure_ascii=False)}
                else:  # openai
                    sse_obj = chunk_to_openai_sse(egress_state, chunk)
                    if sse_obj is not None:
                        yield {"data": json.dumps(sse_obj, ensure_ascii=False)}
                    if isinstance(chunk, ChunkMessageStop):
                        # OpenAI terminator
                        yield {"data": "[DONE]"}

                if isinstance(chunk, ChunkError):
                    return
        except Exception as e:  # noqa: BLE001
            log.exception("stream pipeline crashed")
            if ingress == "anthropic":
                yield {
                    "event": "error",
                    "data": json.dumps(
                        {"type": "error", "error": {"type": "proxy_error", "message": str(e)}}
                    ),
                }
            else:
                yield {
                    "data": json.dumps(
                        {"error": {"type": "proxy_error", "message": str(e)}}
                    )
                }
        finally:
            if session is not None and collected:
                asst = assistant_message_from_chunks(collected)
                if asst is not None:
                    session.schedule(asst)

    return EventSourceResponse(event_generator(), headers=_response_headers(virt_report))


async def _aggregate_response(
    provider: Provider,
    req: CortexRequest,
    api_key: str,
    extra_headers: dict[str, str],
    session: Session | None,
    virt_report: VirtualizationReport | None,
    ingress: str,
) -> JSONResponse:
    upstream_req = req.model_copy(update={"stream": True})
    chunks: list[CortexChunk] = []
    async for chunk in provider.stream(upstream_req, api_key=api_key, extra_headers=extra_headers):
        chunks.append(chunk)
        if isinstance(chunk, ChunkError):
            if ingress == "anthropic":
                content = {
                    "type": "error",
                    "error": {"type": chunk.error_type, "message": chunk.message},
                }
            else:
                content = {
                    "error": {"type": chunk.error_type, "message": chunk.message}
                }
            return JSONResponse(
                status_code=502,
                content=content,
                headers=_response_headers(virt_report),
            )
    if session is not None:
        asst = assistant_message_from_chunks(chunks)
        if asst is not None:
            session.schedule(asst)
    if ingress == "anthropic":
        body = response_from_chunks(chunks, req.model)
    else:
        body = openai_response_from_chunks(chunks, req.model)
    return JSONResponse(content=body, headers=_response_headers(virt_report))


def _response_headers(virt_report: VirtualizationReport | None) -> dict[str, str]:
    headers = {"X-Cortex-Proxy": "anthropic"}
    if virt_report is not None:
        headers["X-Cortex-Virtualized"] = "true" if virt_report.cold_group_count > 0 else "false"
        headers["X-Cortex-Original-Messages"] = str(virt_report.original_message_count)
        headers["X-Cortex-Kept-Messages"] = str(virt_report.kept_message_count)
        headers["X-Cortex-Recap-Tokens"] = str(virt_report.recap_token_estimate)
        if virt_report.degraded:
            headers["X-Cortex-Degraded"] = "virtualize-skipped"
    return headers


# ---------- Helpers ----------


def _try_extract_api_key(request: Request) -> str | None:
    """Like `_extract_api_key` but returns None instead of raising."""
    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def _extract_api_key(request: Request) -> str:
    """Pull the provider API key from the request.

    Anthropic clients send `x-api-key`. OpenAI clients send
    `Authorization: Bearer ...`. We accept either so the same client can speak
    multiple providers without rewiring auth.
    """
    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    raise HTTPException(
        status_code=401,
        detail="missing api key — set x-api-key or Authorization: Bearer ...",
    )


def _passthrough_headers(request: Request) -> dict[str, str]:
    """Allow-list of upstream-relevant headers we forward as-is.

    We do NOT forward Authorization/x-api-key here — those are handled per
    provider in `_extract_api_key`. We strip cortex-private headers.
    """
    allow = {
        "anthropic-beta",
        "openai-organization",
        "openai-project",
        "openai-beta",
        "user-agent",
    }
    out: dict[str, str] = {}
    for k, v in request.headers.items():
        if k.lower() in allow:
            out[k.lower()] = v
    return out


def _bool_header(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


# ---------- Module-level app + entry point ----------

app = _build_app()


def main() -> None:
    """Console entry point. `python -m cortex.server` runs this."""
    import uvicorn

    s = get_cortex_settings()
    uvicorn.run(
        "cortex.server:app",
        host=s.host,
        port=s.port,
        log_level=s.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
