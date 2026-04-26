import time
import uuid
from typing import AsyncIterator, Dict, Optional, Tuple

import httpx

from free_llm_router.catalog import ModelSpec, ProviderSpec
from free_llm_router.schemas import ChatCompletionRequest, ChatCompletionResponse


class ProviderClient:
    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds

    async def send_chat_completion(
        self,
        request: ChatCompletionRequest,
        model: ModelSpec,
    ) -> Tuple[ChatCompletionResponse, float]:
        started = time.time()
        provider = model.provider
        if provider is None:
            raise RuntimeError("Model provider missing.")

        adapter = provider.adapter
        if adapter == "openai":
            response = await self._openai_chat(request, provider, model)
        else:
            raise RuntimeError("Unsupported adapter: {0}".format(adapter))
        return response, (time.time() - started) * 1000.0

    async def stream_openai_chat(
        self,
        request: ChatCompletionRequest,
        model: ModelSpec,
    ) -> AsyncIterator[bytes]:
        provider = model.provider
        if provider is None or provider.adapter != "openai":
            raise RuntimeError("Streaming is currently supported only for OpenAI-compatible providers.")

        payload = self._request_payload(request, model.id)
        headers = self._headers(provider)
        headers.update(request.extra_headers or {})
        async with httpx.AsyncClient(timeout=self.timeout_seconds, trust_env=False) as client:
            async with client.stream(
                "POST",
                "{0}/chat/completions".format(provider.resolved_base_url().rstrip("/")),
                json=payload,
                headers=headers,
            ) as response:
                await self._raise_for_status(response, provider.id, model.id)
                async for chunk in response.aiter_bytes():
                    yield chunk

    async def healthcheck(self, model: ModelSpec) -> Tuple[bool, Optional[int], Optional[float], Optional[str]]:
        provider = model.provider
        if provider is None:
            return False, None, None, "Missing provider"

        started = time.time()
        try:
            async with httpx.AsyncClient(timeout=min(self.timeout_seconds, 20), trust_env=False) as client:
                if provider.adapter == "openai":
                    return await self._openai_healthcheck(client, provider, model, started)

        except Exception as exc:
            return False, None, (time.time() - started) * 1000.0, str(exc)

        return False, None, None, "Unknown adapter"

    async def _openai_chat(
        self,
        request: ChatCompletionRequest,
        provider: ProviderSpec,
        model: ModelSpec,
    ) -> ChatCompletionResponse:
        payload = self._request_payload(request, model.id)
        headers = self._headers(provider)
        headers.update(request.extra_headers or {})
        async with httpx.AsyncClient(timeout=self.timeout_seconds, trust_env=False) as client:
            response = await client.post(
                "{0}/chat/completions".format(provider.resolved_base_url().rstrip("/")),
                json=payload,
                headers=headers,
            )
            await self._raise_for_status(response, provider.id, model.id)
            data = response.json()

        return ChatCompletionResponse(
            id=data.get("id", "chatcmpl-{0}".format(uuid.uuid4().hex[:12])),
            created=int(data.get("created", time.time())),
            model=data.get("model", model.id),
            choices=data.get("choices", []),
            usage=data.get("usage", {}),
            router={},
        )

    async def _openai_healthcheck(
        self,
        client: httpx.AsyncClient,
        provider: ProviderSpec,
        model: ModelSpec,
        started: float,
    ) -> Tuple[bool, Optional[int], Optional[float], Optional[str]]:
        base_url = provider.resolved_base_url().rstrip("/")
        headers = self._headers(provider)
        response = await client.get("{0}/models".format(base_url), headers=headers)
        latency_ms = (time.time() - started) * 1000.0
        if response.is_success:
            return True, response.status_code, latency_ms, None
        if response.status_code not in {404, 405}:
            return False, response.status_code, latency_ms, response.text[:200]

        tiny_payload = {
            "model": model.id,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 4,
        }
        response = await client.post(
            "{0}/chat/completions".format(base_url),
            headers=headers,
            json=tiny_payload,
        )
        latency_ms = (time.time() - started) * 1000.0
        return response.is_success, response.status_code, latency_ms, None if response.is_success else response.text[:200]

    async def _raise_for_status(self, response: httpx.Response, provider_id: str, model_id: str) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = response.text.strip()
            message = "{0} provider={1} model={2}".format(str(exc), provider_id, model_id)
            if body:
                message = "{0} body={1}".format(message, body[:1000])
            raise RuntimeError(message) from exc

    def _headers(self, provider: ProviderSpec) -> Dict[str, str]:
        if provider.api_key_env and not provider.api_key():
            raise RuntimeError(
                "Missing required environment variable {0} for provider {1}".format(
                    provider.api_key_env,
                    provider.id,
                )
            )
        headers = {"Content-Type": "application/json"}
        api_key = provider.api_key()
        if api_key:
            headers["Authorization"] = "Bearer {0}".format(api_key)
        if provider.id == "github_models":
            headers["Accept"] = "application/vnd.github+json"
            headers["X-GitHub-Api-Version"] = "2026-03-10"
        headers.update(provider.extra_headers)
        return headers

    def _request_payload(self, request: ChatCompletionRequest, model_id: str) -> Dict[str, object]:
        payload = request.dict(
            exclude_none=True,
            exclude={"router", "extra_headers"},
        )
        payload["model"] = model_id
        return payload
