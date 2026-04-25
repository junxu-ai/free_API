from typing import Dict, List

import yaml

from free_llm_router.catalog_seed import CATALOG


def _prompt(text: str, default: str = "") -> str:
    suffix = " [{0}]".format(default) if default else ""
    value = input("{0}{1}: ".format(text, suffix)).strip()
    return value or default


def _catalog_provider_models(provider_id: str) -> List[dict]:
    return [model for model in CATALOG["models"] if model["provider_id"] == provider_id]


def _recommended_models(provider_id: str) -> List[str]:
    models = _catalog_provider_models(provider_id)
    models.sort(
        key=lambda item: (item["quality_score"], item["context_length"], item["speed_score"]),
        reverse=True,
    )
    return [model["id"] for model in models[:3]]


def _print_provider_guide(provider: Dict[str, object]) -> None:
    print("  Reference: {0}".format(provider.get("setup_reference", "catalog seed")))
    print("  Docs: {0}".format(provider.get("docs_url", "")))
    print("  Base URL: {0}".format(provider.get("base_url", "")))
    if provider.get("auth_hint"):
        print("  Auth: {0}".format(provider["auth_hint"]))
    if provider.get("environment_variables"):
        print("  Env vars: {0}".format(", ".join(provider["environment_variables"])))
    if provider.get("example_model"):
        print("  Example model: {0}".format(provider["example_model"]))
    if provider.get("key_steps"):
        print("  Key setup:")
        for step in provider["key_steps"]:
            print("    - {0}".format(step))
    if provider.get("notes"):
        print("  Notes:")
        for note in provider["notes"]:
            print("    - {0}".format(note))


def _select_providers() -> List[dict]:
    providers = CATALOG["providers"]
    selected = []
    print("\nAvailable free-tier providers in the seed catalog:\n")
    for index, provider in enumerate(providers, start=1):
        print("{0}. {1} ({2}) [{3}]".format(index, provider["name"], provider["id"], provider["setup_reference"]))

    raw_selection = _prompt("\nEnter provider numbers separated by commas", "7,8,11")
    indices = []
    for item in raw_selection.split(","):
        item = item.strip()
        if item.isdigit():
            indices.append(int(item) - 1)

    for index in indices:
        if index < 0 or index >= len(providers):
            continue
        provider = providers[index]
        provider_models = _catalog_provider_models(provider["id"])
        recommended = _recommended_models(provider["id"])
        print("\nConfiguring {0}".format(provider["name"]))
        _print_provider_guide(provider)
        if provider_models:
            print("  Available models:")
            for model in provider_models:
                print(
                    "    - {0} [{1}, {2}, {3} ctx, {4}]".format(
                        model["id"],
                        model["performance_tier"],
                        ",".join(model["scenarios"]),
                        model["context_length"],
                        model["rate_limit"],
                    )
                )

        api_key_env = _prompt("  Environment variable name for API key", provider.get("api_key_env", ""))
        account_id_env = ""
        if provider.get("account_id_env"):
            account_id_env = _prompt("  Environment variable name for account id", provider["account_id_env"])

        default_allowlist = ",".join(recommended)
        allowlist = _prompt("  Comma-separated model allowlist", default_allowlist)
        entry = {
            "id": provider["id"],
            "enabled": True,
        }
        if api_key_env:
            entry["api_key_env"] = api_key_env
        if account_id_env:
            entry["account_id_env"] = account_id_env
        if allowlist:
            entry["model_allowlist"] = [item.strip() for item in allowlist.split(",") if item.strip()]
        selected.append(entry)
    return selected


def run_wizard(output_path: str) -> None:
    print("Free LLM Router setup wizard")
    print("This writes provider references and env var names, never raw secrets.\n")

    host = _prompt("API host", "0.0.0.0")
    port = int(_prompt("API port", "8000"))
    sqlite_path = _prompt("SQLite path", "data/router.db")
    health_interval = int(_prompt("Health check interval seconds", "300"))
    default_performance = _prompt("Default performance tier", "high")

    providers = _select_providers()
    if not providers:
        raise RuntimeError("At least one provider must be selected.")

    config = {
        "app": {
            "host": host,
            "port": port,
            "request_timeout_seconds": 60,
            "health_check_interval_seconds": health_interval,
            "sqlite_path": sqlite_path,
        },
        "router": {
            "default_model": "auto",
            "default_scenario": "generation",
            "default_performance": default_performance,
            "fallback_order": ["high", "medium", "low"],
        },
        "providers": providers,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    print("\nWrote configuration to {0}".format(output_path))
    print("Next steps:")
    print("  1. Export the environment variables shown in the wizard for each provider.")
    print("  2. Review `/v1/providers` after startup for the normalized provider guide view.")
    print("  3. Start the API with `python -m free_llm_router serve --config {0}`.".format(output_path))
