import unittest
import os
import sys
if __package__ in {None, ""}:
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if package_root not in sys.path:
        sys.path.insert(0, package_root)


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
                    "qwen/qwen3-coder:free",
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
            router=RouterHints(model_ids=["qwen/qwen3-coder:free"]),
        )
        plan = router.build_plan(request)
        self.assertEqual(plan.candidates[0].id, "qwen/qwen3-coder:free")

    def test_provider_base_url_template_resolves_account_id(self) -> None:
        os.environ["CLOUDFLARE_ACCOUNT_ID"] = "acct-123"
        settings = Settings(
            providers=[
                ProviderConfig(
                    id="cloudflare",
                    enabled=True,
                    account_id_env="CLOUDFLARE_ACCOUNT_ID",
                )
            ]
        )
        catalog = Catalog(settings)
        provider = catalog.providers["cloudflare"]
        self.assertEqual(
            provider.resolved_base_url(),
            "https://api.cloudflare.com/client/v4/accounts/acct-123/ai/v1",
        )


if __name__ == "__main__":
    unittest.main()
