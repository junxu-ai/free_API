from dataclasses import dataclass
from typing import Dict, List, Optional

from free_llm_router.catalog import Catalog, ModelSpec
from free_llm_router.scenario import classify_scenario, flatten_messages
from free_llm_router.schemas import ChatCompletionRequest


PERFORMANCE_SCORES = {"high": 3.0, "medium": 2.0, "low": 1.0}


@dataclass
class RoutingPlan:
    scenario: str
    performance: str
    prompt_length: int
    candidates: List[ModelSpec]


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


class ModelRouter:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def build_plan(
        self,
        request: ChatCompletionRequest,
        health_snapshot: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> RoutingPlan:
        health_snapshot = health_snapshot or {}
        prompt_text = flatten_messages(request.messages)
        prompt_tokens = estimate_tokens(prompt_text)
        scenario = self._resolve_scenario(request)
        performance = self._resolve_performance(request)

        explicit_model = self.catalog.get_model(request.model)
        if explicit_model and explicit_model.provider and explicit_model.provider.enabled:
            return RoutingPlan(
                scenario=scenario,
                performance=performance,
                prompt_length=prompt_tokens,
                candidates=[explicit_model],
            )

        candidates = []
        for model in self.catalog.active_models():
            if request.router and request.router.provider_ids and model.provider_id not in request.router.provider_ids:
                continue
            if request.router and request.router.model_ids and model.id not in request.router.model_ids:
                continue
            if request.router and request.router.minimum_context_tokens and model.context_length < request.router.minimum_context_tokens:
                continue
            if model.context_length < prompt_tokens:
                continue
            score = self._score(model, scenario, performance, prompt_tokens, health_snapshot.get(model.id, {}))
            candidates.append((score, model))

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = [model for _, model in candidates]

        if not selected:
            fallback = sorted(
                self.catalog.active_models(),
                key=lambda model: (model.context_length, model.quality_score),
                reverse=True,
            )
            selected = fallback[:5]

        return RoutingPlan(
            scenario=scenario,
            performance=performance,
            prompt_length=prompt_tokens,
            candidates=selected[:8],
        )

    def _resolve_scenario(self, request: ChatCompletionRequest) -> str:
        if request.router and request.router.scenario:
            return request.router.scenario
        if request.model.startswith("auto:"):
            parts = request.model.split(":")
            if parts[-1] in {"generation", "reasoning", "agentic", "coding", "summarization", "vision"}:
                return parts[-1]
        return classify_scenario(request.messages)

    def _resolve_performance(self, request: ChatCompletionRequest) -> str:
        if request.router and request.router.performance:
            return request.router.performance
        parts = request.model.split(":")
        for part in parts:
            if part in PERFORMANCE_SCORES:
                return part
        return self.catalog.settings.router.default_performance

    def _score(
        self,
        model: ModelSpec,
        scenario: str,
        performance: str,
        prompt_tokens: int,
        health: Dict[str, float],
    ) -> float:
        score = model.quality_score + (model.speed_score * 0.2)
        score += PERFORMANCE_SCORES.get(model.performance_tier, 0.0)

        if scenario in model.scenarios:
            score += 3.5
        if performance == model.performance_tier:
            score += 2.5

        context_ratio = min(4.0, float(model.context_length) / max(prompt_tokens, 1) / 2.0)
        score += context_ratio

        success_rate = float(health.get("success_rate", 0.75))
        latency_ms = float(health.get("latency_ms", 1500.0))
        score += success_rate * 4.0
        score -= min(4.0, latency_ms / 1000.0)
        return score
