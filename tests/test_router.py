import unittest

from free_llm_router.catalog import Catalog
from free_llm_router.config import ProviderConfig, Settings
from free_llm_router.router import ModelRouter
from free_llm_router.schemas import ChatCompletionRequest, ChatMessage, RouterHints


def build_catalog() -> Catalog:
    settings = Settings(
        providers=[
            ProviderConfig(
                id="groq",
                enabled=True,
                model_allowlist=[
                    "llama-3.3-70b-versatile",
                    "deepseek-r1-distill-70b",
                    "qwen3-32b",
                ],
            ),
            ProviderConfig(
                id="openrouter",
                enabled=True,
                model_allowlist=[
                    "deepseek/deepseek-r1-0528:free",
                    "qwen/qwen3-coder-480b-a35b:free",
                ],
            ),
        ]
    )
    return Catalog(settings)


class RouterTests(unittest.TestCase):
    def test_reasoning_prefers_reasoning_models(self) -> None:
        router = ModelRouter(build_catalog())
        request = ChatCompletionRequest(
            model="auto",
            messages=[ChatMessage(role="user", content="Reason step by step about this number theory puzzle.")],
        )
        plan = router.build_plan(request)
        self.assertTrue(plan.candidates)
        self.assertIn("reasoning", plan.candidates[0].scenarios)

    def test_router_hint_filters_models(self) -> None:
        router = ModelRouter(build_catalog())
        request = ChatCompletionRequest(
            model="auto",
            messages=[ChatMessage(role="user", content="Write Python code to parse a CSV file.")],
            router=RouterHints(model_ids=["qwen/qwen3-coder-480b-a35b:free"]),
        )
        plan = router.build_plan(request)
        self.assertEqual(plan.candidates[0].id, "qwen/qwen3-coder-480b-a35b:free")


if __name__ == "__main__":
    unittest.main()
