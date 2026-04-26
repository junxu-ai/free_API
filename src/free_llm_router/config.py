import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    request_timeout_seconds: int = 60
    health_check_interval_seconds: int = 300
    sqlite_path: str = "data/router.db"


@dataclass
class RouterConfig:
    default_model: str = "auto"
    default_scenario: str = "generation"
    default_performance: str = "high"
    fallback_order: List[str] = field(default_factory=lambda: ["high", "medium", "low"])


@dataclass
class ProviderConfig:
    id: str
    enabled: bool = True
    api_key_env: Optional[str] = None
    account_id_env: Optional[str] = None
    base_url_override: Optional[str] = None
    model_allowlist: List[str] = field(default_factory=list)
    model_blocklist: List[str] = field(default_factory=list)
    default_model: Optional[str] = None
    extra_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class Settings:
    app: AppConfig = field(default_factory=AppConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    providers: List[ProviderConfig] = field(default_factory=list)

    def provider_map(self) -> Dict[str, ProviderConfig]:
        return {provider.id: provider for provider in self.providers}


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def _resolve_base_dir(config_path: str) -> str:
    config_dir = os.path.dirname(os.path.abspath(config_path))
    if os.path.basename(config_dir).lower() == "config":
        return os.path.dirname(config_dir)
    return config_dir


def _resolve_path(value: str, base_dir: str) -> str:
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(base_dir, value))


def _make_app_config(data: Dict[str, Any]) -> AppConfig:
    return AppConfig(**(data or {}))


def _make_router_config(data: Dict[str, Any]) -> RouterConfig:
    return RouterConfig(**(data or {}))


def _make_providers(items: List[Dict[str, Any]]) -> List[ProviderConfig]:
    return [ProviderConfig(**item) for item in items or []]


def load_settings(path: str) -> Settings:
    config_path = os.path.abspath(path)
    base_dir = _resolve_base_dir(config_path)
    raw = _read_yaml(path)
    settings = Settings(
        app=_make_app_config(raw.get("app", {})),
        router=_make_router_config(raw.get("router", {})),
        providers=_make_providers(raw.get("providers", [])),
    )
    settings.app.sqlite_path = _resolve_path(settings.app.sqlite_path, base_dir)
    sqlite_dir = os.path.dirname(settings.app.sqlite_path)
    if sqlite_dir:
        os.makedirs(sqlite_dir, exist_ok=True)
    return settings
