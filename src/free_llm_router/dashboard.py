import argparse
import os
from typing import Dict, List, Optional

import httpx
import streamlit as st

from free_llm_router.catalog import Catalog
from free_llm_router.config import load_settings
from free_llm_router.store import RouterStore


CHAT_STATE_KEY = "chat_tester_messages"
CHAT_META_KEY = "chat_tester_metadata"


def _parse_args() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=os.getenv("FREE_LLM_ROUTER_CONFIG", "config/config.yaml"))
    args, _ = parser.parse_known_args()
    return args.config


def _metrics_row(summary: Dict[str, object]) -> Dict[str, int]:
    totals = summary.get("totals", {})
    return {
        "requests": int(totals.get("requests") or 0),
        "successful_requests": int(totals.get("successful_requests") or 0),
        "total_tokens": int(totals.get("total_tokens") or 0),
    }


def _chart_data(rows: List[Dict[str, object]]) -> Dict[str, int]:
    return {row["model_id"]: int(row.get("requests") or 0) for row in rows[:10]}


def _default_api_base(settings_host: str, settings_port: int) -> str:
    host = settings_host
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return "http://{0}:{1}/v1".format(host, settings_port)


def _ensure_chat_state() -> None:
    if CHAT_STATE_KEY not in st.session_state:
        st.session_state[CHAT_STATE_KEY] = []
    if CHAT_META_KEY not in st.session_state:
        st.session_state[CHAT_META_KEY] = []


def _clear_chat() -> None:
    st.session_state[CHAT_STATE_KEY] = []
    st.session_state[CHAT_META_KEY] = []


def _extract_content(data: Dict[str, object]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return str(content)


def _selected_provider_hint(selected_provider: str) -> Optional[List[str]]:
    if selected_provider == "all":
        return None
    return [selected_provider]


def _post_chat_completion(
    api_base_url: str,
    messages: List[Dict[str, str]],
    model: str,
    selected_provider: str,
    scenario: str,
    performance: str,
    timeout_seconds: int,
) -> Dict[str, object]:
    router_hints: Dict[str, object] = {}
    provider_ids = _selected_provider_hint(selected_provider)
    if provider_ids:
        router_hints["provider_ids"] = provider_ids
    if scenario != "auto":
        router_hints["scenario"] = scenario
    if performance != "auto":
        router_hints["performance"] = performance

    payload: Dict[str, object] = {
        "model": model,
        "messages": messages,
    }
    if router_hints:
        payload["router"] = router_hints

    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(
            "{0}/chat/completions".format(api_base_url.rstrip("/")),
            json=payload,
        )
        response.raise_for_status()
        return response.json()


def _render_overview(catalog: Catalog, store: RouterStore) -> None:
    summary = store.usage_summary()
    metrics = _metrics_row(summary)

    col1, col2, col3 = st.columns(3)
    col1.metric("Requests", metrics["requests"])
    col2.metric("Successful", metrics["successful_requests"])
    col3.metric("Tokens", metrics["total_tokens"])

    st.subheader("Configured Providers")
    provider_rows = []
    for provider in catalog.providers.values():
        try:
            base_url = provider.resolved_base_url()
        except RuntimeError:
            base_url = provider.base_url
        provider_rows.append(
            {
                "provider_id": provider.id,
                "name": provider.name,
                "enabled": provider.enabled,
                "adapter": provider.adapter,
                "base_url": base_url,
                "auth_hint": provider.auth_hint,
                "env_vars": ", ".join(provider.environment_variables),
                "example_model": provider.example_model,
                "setup_reference": provider.setup_reference,
            }
        )
    st.dataframe(provider_rows, use_container_width=True)

    st.subheader("Model Inventory")
    model_rows = []
    health = store.health_snapshot()
    for model in catalog.active_models():
        snapshot = health.get(model.id, {})
        model_rows.append(
            {
                "model_id": model.id,
                "provider_id": model.provider_id,
                "tier": model.performance_tier,
                "context": model.context_length,
                "scenarios": ", ".join(model.scenarios),
                "healthy": (snapshot.get("success_rate", 0.0) >= 0.5),
                "latency_ms": round(snapshot.get("latency_ms", 0.0), 2),
                "rate_limit": model.rate_limit,
            }
        )
    st.dataframe(model_rows, use_container_width=True)

    st.subheader("Latest Health Checks")
    st.dataframe(store.provider_status_rows(), use_container_width=True)

    st.subheader("Usage by Model")
    chart_values = _chart_data(summary.get("by_model", []))
    if chart_values:
        st.bar_chart(chart_values)
    else:
        st.info("No requests logged yet.")

    st.subheader("Recent Requests")
    st.dataframe(summary.get("recent_requests", []), use_container_width=True)


def _render_chat_tester(catalog: Catalog, settings_timeout: int, default_api_base: str) -> None:
    _ensure_chat_state()

    st.subheader("Chat Tester")
    st.caption("Use the local router endpoint to test specific providers, models, or auto routing.")

    active_providers = catalog.active_providers()
    active_models = catalog.active_models()
    provider_options = ["all"] + [provider.id for provider in active_providers]
    model_options = ["auto"] + [model.id for model in active_models]

    control_col1, control_col2 = st.columns([3, 1])
    with control_col1:
        api_base_url = st.text_input(
            "Router API base URL",
            value=st.session_state.get("chat_api_base_url", default_api_base),
            help="Point this at the running FastAPI router, usually the local `/v1` endpoint.",
            key="chat_api_base_url",
        )
    with control_col2:
        st.write("")
        if st.button("Clear Chat", use_container_width=True):
            _clear_chat()
            st.rerun()

    selector_col1, selector_col2, selector_col3, selector_col4 = st.columns(4)
    with selector_col1:
        selected_provider = st.selectbox(
            "Provider",
            options=provider_options,
            format_func=lambda item: "All providers" if item == "all" else item,
            key="chat_selected_provider",
        )
    with selector_col2:
        available_models = ["auto"]
        for model in active_models:
            if selected_provider == "all" or model.provider_id == selected_provider:
                available_models.append(model.id)
        selected_model = st.selectbox("Model", options=available_models, key="chat_selected_model")
    with selector_col3:
        selected_scenario = st.selectbox(
            "Scenario Hint",
            options=["auto", "generation", "reasoning", "agentic", "coding", "summarization", "vision"],
            key="chat_selected_scenario",
        )
    with selector_col4:
        selected_performance = st.selectbox(
            "Performance Hint",
            options=["auto", "high", "medium", "low"],
            key="chat_selected_performance",
        )

    if selected_model != "auto":
        chosen_model = catalog.get_model(selected_model)
        if chosen_model:
            st.caption(
                "Testing `{0}` on provider `{1}` with {2} context tokens.".format(
                    chosen_model.id,
                    chosen_model.provider_id,
                    chosen_model.context_length,
                )
            )
    elif selected_provider != "all":
        st.caption("`auto` routing is restricted to provider `{0}`.".format(selected_provider))

    for index, message in enumerate(st.session_state[CHAT_STATE_KEY]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                meta = st.session_state[CHAT_META_KEY][index]
                router_meta = meta.get("router") or {}
                if router_meta:
                    st.caption(
                        "Provider: `{0}` | Model: `{1}` | Scenario: `{2}` | Performance: `{3}`".format(
                            router_meta.get("selected_provider", "n/a"),
                            router_meta.get("selected_model", "n/a"),
                            router_meta.get("scenario", "n/a"),
                            router_meta.get("performance", "n/a"),
                        )
                    )

    user_prompt = st.chat_input("Send a test prompt to the router")
    if not user_prompt:
        return

    st.session_state[CHAT_STATE_KEY].append({"role": "user", "content": user_prompt})
    st.session_state[CHAT_META_KEY].append({})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    api_messages = list(st.session_state[CHAT_STATE_KEY])
    with st.chat_message("assistant"):
        with st.spinner("Calling router..."):
            try:
                data = _post_chat_completion(
                    api_base_url=api_base_url,
                    messages=api_messages,
                    model=selected_model,
                    selected_provider=selected_provider,
                    scenario=selected_scenario,
                    performance=selected_performance,
                    timeout_seconds=settings_timeout,
                )
                assistant_text = _extract_content(data)
                router_meta = data.get("router") or {}
                st.markdown(assistant_text or "_Empty response_")
                if router_meta:
                    st.caption(
                        "Provider: `{0}` | Model: `{1}` | Scenario: `{2}` | Performance: `{3}`".format(
                            router_meta.get("selected_provider", "n/a"),
                            router_meta.get("selected_model", "n/a"),
                            router_meta.get("scenario", "n/a"),
                            router_meta.get("performance", "n/a"),
                        )
                    )
                with st.expander("Raw Response"):
                    st.json(data)
                st.session_state[CHAT_STATE_KEY].append({"role": "assistant", "content": assistant_text or ""})
                st.session_state[CHAT_META_KEY].append(data)
            except Exception as exc:
                error_message = "Request failed: {0}".format(exc)
                st.error(error_message)
                st.session_state[CHAT_STATE_KEY].append({"role": "assistant", "content": error_message})
                st.session_state[CHAT_META_KEY].append({"error": str(exc)})


def main() -> None:
    config_path = _parse_args()
    settings = load_settings(config_path)
    catalog = Catalog(settings)
    store = RouterStore(settings.app.sqlite_path)
    default_api_base = _default_api_base(settings.app.host, settings.app.port)

    st.set_page_config(page_title="Free LLM Router Dashboard", layout="wide")
    st.title("Free LLM Router Dashboard")
    st.caption("Status, usage, configured routing inventory, and live API testing.")

    overview_tab, chat_tab = st.tabs(["Overview", "Chat Tester"])

    with overview_tab:
        _render_overview(catalog, store)

    with chat_tab:
        _render_chat_tester(
            catalog=catalog,
            settings_timeout=settings.app.request_timeout_seconds,
            default_api_base=default_api_base,
        )


if __name__ == "__main__":
    main()
