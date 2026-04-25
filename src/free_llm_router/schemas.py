from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


PYDANTIC_V2 = hasattr(BaseModel, "model_fields")


class RouterBaseModel(BaseModel):
    if PYDANTIC_V2:
        model_config = {"protected_namespaces": ()}
    else:
        class Config:
            protected_namespaces = ()


class ChatMessage(BaseModel):
    role: str
    content: Any


class RouterHints(RouterBaseModel):
    scenario: Optional[str] = None
    performance: Optional[str] = None
    minimum_context_tokens: Optional[int] = None
    provider_ids: Optional[List[str]] = None
    model_ids: Optional[List[str]] = None


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: bool = False
    user: Optional[str] = None
    router: Optional[RouterHints] = None
    extra_headers: Dict[str, str] = Field(default_factory=dict)


class ChatCompletionResponseMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionResponseMessage
    finish_reason: str = "stop"


class CompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: CompletionUsage
    router: Dict[str, Any] = Field(default_factory=dict)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    provider_id: str
    performance_tier: str
    context_length: int
    scenarios: List[str]
    rate_limit: str
    healthy: bool = False
    latency_ms: Optional[float] = None


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelCard]


class ProviderCard(BaseModel):
    id: str
    name: str
    category: str
    adapter: str
    base_url: str
    enabled: bool
    auth_hint: Optional[str] = None
    example_model: Optional[str] = None
    environment_variables: List[str] = Field(default_factory=list)
    key_steps: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    docs_url: Optional[str] = None
    setup_reference: Optional[str] = None


class ProviderListResponse(BaseModel):
    object: str = "list"
    data: List[ProviderCard]
