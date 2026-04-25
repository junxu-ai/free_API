import argparse
import os
from typing import Dict, List

import streamlit as st

from free_llm_router.catalog import Catalog
from free_llm_router.config import load_settings
from free_llm_router.store import RouterStore


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


def main() -> None:
    config_path = _parse_args()
    settings = load_settings(config_path)
    catalog = Catalog(settings)
    store = RouterStore(settings.app.sqlite_path)

    st.set_page_config(page_title="Free LLM Router Dashboard", layout="wide")
    st.title("Free LLM Router Dashboard")
    st.caption("Status, usage, and configured routing inventory.")

    summary = store.usage_summary()
    metrics = _metrics_row(summary)

    col1, col2, col3 = st.columns(3)
    col1.metric("Requests", metrics["requests"])
    col2.metric("Successful", metrics["successful_requests"])
    col3.metric("Tokens", metrics["total_tokens"])

    st.subheader("Configured Providers")
    provider_rows = []
    for provider in catalog.active_providers():
        provider_rows.append(
            {
                "provider_id": provider.id,
                "name": provider.name,
                "adapter": provider.adapter,
                "base_url": provider.base_url,
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


if __name__ == "__main__":
    main()
