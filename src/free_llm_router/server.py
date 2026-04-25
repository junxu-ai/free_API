import os
import time
import uuid
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from free_llm_router.catalog import Catalog
from free_llm_router.clients import ProviderClient
from free_llm_router.config import Settings, load_settings
from free_llm_router.health import HealthMonitor
from free_llm_router.router import ModelRouter
from free_llm_router.schemas import ChatCompletionRequest, ChatCompletionResponse, ModelCard, ModelListResponse
from free_llm_router.store import RouterStore


def create_app(config_path: str) -> FastAPI:
    settings = load_settings(config_path)
    catalog = Catalog(settings)
    store = RouterStore(settings.app.sqlite_path)
    client = ProviderClient(settings.app.request_timeout_seconds)
    router = ModelRouter(catalog)
    health_monitor = HealthMonitor(
        catalog=catalog,
        store=store,
        client=client,
        interval_seconds=settings.app.health_check_interval_seconds,
    )

    app = FastAPI(title="Free LLM Router", version="0.1.0")

    app.state.settings = settings
    app.state.catalog = catalog
    app.state.store = store
    app.state.client = client
    app.state.router = router
    app.state.health_monitor = health_monitor

    @app.on_event("startup")
    async def on_startup() -> None:
        await health_monitor.start()

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        await health_monitor.stop()

    @app.get("/healthz")
    async def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models() -> ModelListResponse:
        snapshot = store.health_snapshot()
        cards = []
        for model in catalog.active_models():
            health = snapshot.get(model.id, {})
            cards.append(
                ModelCard(
                    id=model.id,
                    provider_id=model.provider_id,
                    performance_tier=model.performance_tier,
                    context_length=model.context_length,
                    scenarios=model.scenarios,
                    rate_limit=model.rate_limit,
                    healthy=(health.get("success_rate", 0.0) >= 0.5),
                    latency_ms=health.get("latency_ms"),
                )
            )
        return ModelListResponse(data=cards)

    @app.get("/v1/router/summary")
    async def router_summary() -> Dict[str, object]:
        return {
            "providers": [provider.id for provider in catalog.active_providers()],
            "models": [model.id for model in catalog.active_models()],
            "usage": store.usage_summary(),
            "health": store.provider_status_rows(),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        health_snapshot = store.health_snapshot()
        plan = router.build_plan(request, health_snapshot)
        if not plan.candidates:
            raise HTTPException(status_code=400, detail="No configured models are available.")

        request_id = "router-{0}".format(uuid.uuid4().hex[:12])

        if request.stream:
            primary = plan.candidates[0]
            try:
                stream = client.stream_openai_chat(request, primary)
                return StreamingResponse(stream, media_type="text/event-stream")
            except Exception as exc:
                raise HTTPException(status_code=502, detail=str(exc))

        last_error = None
        for model in plan.candidates:
            started = time.time()
            try:
                response, latency_ms = await client.send_chat_completion(request, model)
                router_meta = {
                    "scenario": plan.scenario,
                    "performance": plan.performance,
                    "selected_model": model.id,
                    "selected_provider": model.provider_id,
                    "candidate_models": [candidate.id for candidate in plan.candidates],
                }
                response.router = router_meta
                usage = response.usage
                store.log_request(
                    request_id=request_id,
                    provider_id=model.provider_id,
                    model_id=model.id,
                    scenario=plan.scenario,
                    performance=plan.performance,
                    success=True,
                    latency_ms=latency_ms,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                )
                return JSONResponse(content=response.dict())
            except Exception as exc:
                last_error = str(exc)
                latency_ms = (time.time() - started) * 1000.0
                store.log_request(
                    request_id=request_id,
                    provider_id=model.provider_id,
                    model_id=model.id,
                    scenario=plan.scenario,
                    performance=plan.performance,
                    success=False,
                    latency_ms=latency_ms,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    error=last_error,
                )

        raise HTTPException(status_code=502, detail=last_error or "All candidate models failed.")

    return app


def resolve_config_path(config_path: str = None) -> str:
    if config_path:
        return config_path
    env_path = os.getenv("FREE_LLM_ROUTER_CONFIG")
    if env_path:
        return env_path
    return "config/config.yaml"


def load_app(config_path: str = None) -> FastAPI:
    return create_app(resolve_config_path(config_path))


app = FastAPI(title="Free LLM Router")
