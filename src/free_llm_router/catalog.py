import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from free_llm_router.catalog_seed import CATALOG
from free_llm_router.config import ProviderConfig, Settings


@dataclass
class ProviderSpec:
    id: str
    name: str
    category: str
    adapter: str
    base_url: str
    api_key_env: Optional[str]
    account_id_env: Optional[str]
    docs_url: Optional[str]
    setup_reference: Optional[str]
    auth_hint: Optional[str]
    example_model: Optional[str]
    environment_variables: List[str]
    key_steps: List[str]
    notes: List[str]
    enabled: bool = False
    default_model: Optional[str] = None
    extra_headers: Dict[str, str] = field(default_factory=dict)

    def api_key(self) -> Optional[str]:
        if not self.api_key_env:
            return None
        return os.getenv(self.api_key_env)

    def account_id(self) -> Optional[str]:
        if not self.account_id_env:
            return None
        return os.getenv(self.account_id_env)

    def resolved_base_url(self) -> str:
        base_url = self.base_url
        if "{account_id}" in base_url:
            account_id = self.account_id()
            if not account_id:
                raise RuntimeError(
                    "Provider {0} requires {1} to resolve the base URL.".format(
                        self.id,
                        self.account_id_env,
                    )
                )
            base_url = base_url.replace("{account_id}", account_id)
        return base_url


@dataclass
class ModelSpec:
    id: str
    provider_id: str
    context_length: int
    max_output_tokens: int
    performance_tier: str
    scenarios: List[str]
    quality_score: float
    speed_score: float
    rate_limit: str
    modality: str
    provider: Optional[ProviderSpec] = None


class Catalog:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.providers: Dict[str, ProviderSpec] = {}
        self.models: Dict[str, ModelSpec] = {}
        self._load()

    def _load(self) -> None:
        provider_overrides = self.settings.provider_map()
        for item in CATALOG["providers"]:
            override = provider_overrides.get(item["id"])
            provider = ProviderSpec(
                id=item["id"],
                name=item["name"],
                category=item["category"],
                adapter=item["adapter"],
                base_url=(override.base_url_override if override and override.base_url_override else item["base_url"]),
                api_key_env=(override.api_key_env if override and override.api_key_env else item.get("api_key_env")),
                account_id_env=(override.account_id_env if override and override.account_id_env else item.get("account_id_env")),
                docs_url=item.get("docs_url"),
                setup_reference=item.get("setup_reference"),
                auth_hint=item.get("auth_hint"),
                example_model=item.get("example_model"),
                environment_variables=list(item.get("environment_variables", [])),
                key_steps=list(item.get("key_steps", [])),
                notes=list(item.get("notes", [])),
                enabled=bool(override and override.enabled),
                default_model=(override.default_model if override else None),
                extra_headers=(override.extra_headers if override else {}),
            )
            self.providers[provider.id] = provider

        for item in CATALOG["models"]:
            provider = self.providers.get(item["provider_id"])
            if not provider:
                continue
            override = provider_overrides.get(provider.id)
            if override and override.model_allowlist and item["id"] not in override.model_allowlist:
                continue
            if override and item["id"] in override.model_blocklist:
                continue
            model = ModelSpec(
                id=item["id"],
                provider_id=item["provider_id"],
                context_length=item["context_length"],
                max_output_tokens=item["max_output_tokens"],
                performance_tier=item["performance_tier"],
                scenarios=list(item["scenarios"]),
                quality_score=float(item["quality_score"]),
                speed_score=float(item["speed_score"]),
                rate_limit=item["rate_limit"],
                modality=item["modality"],
                provider=provider,
            )
            self.models[model.id] = model

    def active_models(self) -> List[ModelSpec]:
        return [model for model in self.models.values() if model.provider and model.provider.enabled]

    def active_providers(self) -> List[ProviderSpec]:
        return [provider for provider in self.providers.values() if provider.enabled]

    def models_for_provider(self, provider_id: str) -> List[ModelSpec]:
        return [model for model in self.active_models() if model.provider_id == provider_id]

    def get_model(self, model_id: str) -> Optional[ModelSpec]:
        return self.models.get(model_id)

    def get_provider_config(self, provider_id: str) -> Optional[ProviderConfig]:
        return self.settings.provider_map().get(provider_id)
