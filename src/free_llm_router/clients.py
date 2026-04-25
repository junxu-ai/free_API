import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, Iterable, Optional, Tuple

import httpx

from free_llm_router.catalog import ModelSpec, ProviderSpec
from free_llm_router.schemas import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, ChatCompletionResponseMessage, CompletionUsage


def _messages_to_prompt(messages: Iterable[Any]) -> str:
    parts = []
    for message in messages:
        content = message.content if hasattr(message, "content") else message.get("content")
        role = message.role if hasattr(message, "role") else message.get("role")
        if isinstance(content, str):
            parts.append("{0}: {1}".format(role, content))
        else:
            parts.append("{0}: {1}".format(role, json.dumps(content)))
    return "\n".join(parts)


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


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
        elif adapter == "cloudflare_workers_ai":
            response = await self._cloudflare_chat(request, provider, model)
        elif adapter == "huggingface_text_generation":
            response = await self._huggingface_chat(request, provider, model)
        elif adapter == "cohere":
            response = await self._cohere_chat(request, provider, model)
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

        payload = request.dict(exclude_none=True)
        payload["model"] = model.id
        headers = self._headers(provider)
        headers.update(request.extra_headers or {})
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            async with client.stream(
                "POST",
                "{0}/chat/completions".format(provider.base_url.rstrip("/")),
                json=payload,
                headers=headers,
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk

    async def healthcheck(self, model: ModelSpec) -> Tuple[bool, Optional[int], Optional[float], Optional[str]]:
        provider = model.provider
        if provider is None:
            return False, None, None, "Missing provider"

        started = time.time()
        try:
            async with httpx.AsyncClient(timeout=min(self.timeout_seconds, 20)) as client:
                if provider.adapter == "openai":
                    response = await client.get(
                        "{0}/models".format(provider.base_url.rstrip("/")),
                        headers=self._headers(provider),
                    )
                    latency_ms = (time.time() - started) * 1000.0
                    return response.is_success, response.status_code, latency_ms, None if response.is_success else response.text[:200]

                if provider.adapter == "cloudflare_workers_ai":
                    account_id = provider.account_id()
                    if not account_id:
                        return False, None, None, "Missing account id env var"
                    response = await client.post(
                        "{0}/accounts/{1}/ai/run/{2}".format(provider.base_url.rstrip("/"), account_id, model.id),
                        headers=self._headers(provider),
                        json={"prompt": "ping"},
                    )
                    latency_ms = (time.time() - started) * 1000.0
                    return response.is_success, response.status_code, latency_ms, None if response.is_success else response.text[:200]

                if provider.adapter == "huggingface_text_generation":
                    response = await client.post(
                        "{0}/models/{1}".format(provider.base_url.rstrip("/"), model.id),
                        headers=self._headers(provider),
                        json={"inputs": "ping", "parameters": {"max_new_tokens": 4}},
                    )
                    latency_ms = (time.time() - started) * 1000.0
                    return response.is_success, response.status_code, latency_ms, None if response.is_success else response.text[:200]

                if provider.adapter == "cohere":
                    response = await client.post(
                        "{0}/chat".format(provider.base_url.rstrip("/")),
                        headers=self._headers(provider),
                        json={"model": model.id, "message": "ping"},
                    )
                    latency_ms = (time.time() - started) * 1000.0
                    return response.is_success, response.status_code, latency_ms, None if response.is_success else response.text[:200]

        except Exception as exc:
            return False, None, (time.time() - started) * 1000.0, str(exc)

        return False, None, None, "Unknown adapter"

    async def _openai_chat(
        self,
        request: ChatCompletionRequest,
        provider: ProviderSpec,
        model: ModelSpec,
    ) -> ChatCompletionResponse:
        payload = request.dict(exclude_none=True)
        payload["model"] = model.id
        headers = self._headers(provider)
        headers.update(request.extra_headers or {})
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                "{0}/chat/completions".format(provider.base_url.rstrip("/")),
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        return ChatCompletionResponse(
            id=data.get("id", "chatcmpl-{0}".format(uuid.uuid4().hex[:12])),
            created=int(data.get("created", time.time())),
            model=data.get("model", model.id),
            choices=data.get("choices", []),
            usage=data.get("usage", {}),
            router={},
        )

    async def _cloudflare_chat(
        self,
        request: ChatCompletionRequest,
        provider: ProviderSpec,
        model: ModelSpec,
    ) -> ChatCompletionResponse:
        account_id = provider.account_id()
        if not account_id:
            raise RuntimeError("Cloudflare provider requires account_id_env to be configured.")
        payload = {
            "messages": [message.dict() for message in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                "{0}/accounts/{1}/ai/run/{2}".format(provider.base_url.rstrip("/"), account_id, model.id),
                json=payload,
                headers=self._headers(provider),
            )
            response.raise_for_status()
            data = response.json()

        result = data.get("result", {})
        content = result.get("response") or result.get("text") or json.dumps(result)
        usage = CompletionUsage(
            prompt_tokens=_estimate_tokens(_messages_to_prompt(request.messages)),
            completion_tokens=_estimate_tokens(content),
            total_tokens=_estimate_tokens(_messages_to_prompt(request.messages)) + _estimate_tokens(content),
        )
        return ChatCompletionResponse(
            id="chatcmpl-{0}".format(uuid.uuid4().hex[:12]),
            created=int(time.time()),
            model=model.id,
            choices=[ChatCompletionChoice(message=ChatCompletionResponseMessage(content=content))],
            usage=usage,
            router={},
        )

    async def _huggingface_chat(
        self,
        request: ChatCompletionRequest,
        provider: ProviderSpec,
        model: ModelSpec,
    ) -> ChatCompletionResponse:
        prompt = _messages_to_prompt(request.messages)
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                "{0}/models/{1}".format(provider.base_url.rstrip("/"), model.id),
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": request.max_tokens or 512,
                        "temperature": request.temperature or 0.7,
                        "return_full_text": False,
                    },
                },
                headers=self._headers(provider),
            )
            response.raise_for_status()
            data = response.json()

        if isinstance(data, list) and data:
            content = data[0].get("generated_text", "")
        elif isinstance(data, dict):
            content = data.get("generated_text", "")
        else:
            content = str(data)

        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(content)
        return ChatCompletionResponse(
            id="chatcmpl-{0}".format(uuid.uuid4().hex[:12]),
            created=int(time.time()),
            model=model.id,
            choices=[ChatCompletionChoice(message=ChatCompletionResponseMessage(content=content))],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            router={},
        )

    async def _cohere_chat(
        self,
        request: ChatCompletionRequest,
        provider: ProviderSpec,
        model: ModelSpec,
    ) -> ChatCompletionResponse:
        prompt = _messages_to_prompt(request.messages)
        payload = {
            "model": model.id,
            "message": prompt,
            "max_tokens": request.max_tokens or 512,
            "temperature": request.temperature or 0.7,
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                "{0}/chat".format(provider.base_url.rstrip("/")),
                json=payload,
                headers=self._headers(provider),
            )
            response.raise_for_status()
            data = response.json()

        message = data.get("message", {})
        content_items = message.get("content") or []
        content = ""
        if content_items and isinstance(content_items, list):
            content = content_items[0].get("text", "")

        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(content)
        return ChatCompletionResponse(
            id="chatcmpl-{0}".format(uuid.uuid4().hex[:12]),
            created=int(time.time()),
            model=model.id,
            choices=[ChatCompletionChoice(message=ChatCompletionResponseMessage(content=content))],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            router={},
        )

    def _headers(self, provider: ProviderSpec) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = provider.api_key()
        if api_key:
            headers["Authorization"] = "Bearer {0}".format(api_key)
        headers.update(provider.extra_headers)
        return headers
