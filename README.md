# Free LLM Router

`free-llm-router` is a Python project that exposes an OpenAI-like API on top of permanent free-tier LLM providers. It is seeded from the provider catalog in [`mnfst/awesome-free-llm-apis`](https://github.com/mnfst/awesome-free-llm-apis) and follows the setup style described in [`free-llm-apis/SKILL.md`](https://github.com/mnfst/awesome-free-llm-apis/blob/main/free-llm-apis/SKILL.md).

## What it does

- Exposes `/v1/chat/completions` and `/v1/models` in an OpenAI-like format.
- Exposes `/v1/providers` to surface provider setup guidance, env vars, and base URLs.
- Lets users pick providers and model allowlists in a local YAML config file.
- Scores configured models by:
  - `performance_tier`: `high`, `medium`, `low`
  - `context_length`
  - `scenarios`: `generation`, `reasoning`, `agentic`, `coding`, `summarization`, `vision`
  - live health and latency from background checks
- Automatically classifies the request scenario when `model="auto"` and routes to the best healthy model.
- Periodically probes providers and stores health and usage data in SQLite.
- Includes an interactive CLI wizard for account configuration.
- Includes a Streamlit dashboard for provider status, usage, and routing inventory.

## Project layout

- `src/free_llm_router/server.py`: FastAPI app and OpenAI-like endpoints
- `src/free_llm_router/router.py`: scenario detection and model selection
- `src/free_llm_router/clients.py`: provider adapters
- `src/free_llm_router/catalog_seed.py`: provider and model metadata aligned with the upstream setup guides
- `src/free_llm_router/store.py`: SQLite usage and health store
- `src/free_llm_router/wizard.py`: interactive config wizard
- `src/free_llm_router/dashboard.py`: Streamlit dashboard
- `config/config.example.yaml`: starter configuration

## Supported provider styles

The seed catalog includes providers from the referenced repository, with working adapters for the most practical free-tier patterns:

- OpenAI-compatible gateways: Groq, Cerebras, GitHub Models, OpenRouter, LLM7.io, Mistral, Gemini OpenAI endpoint, SiliconFlow, Kilo Code, NVIDIA NIM, Zhipu
- Cloudflare Workers AI adapter
- Hugging Face serverless text-generation adapter
- Cohere chat adapter

The catalog is user-editable, so additional providers or models can be added without changing the API surface.

## Quick start

1. Create a config file from the example:

```powershell
Copy-Item config\config.example.yaml config\config.yaml
```

2. Run the setup wizard:

```powershell
D:\Anaconda\python.exe -m free_llm_router wizard --output config\config.yaml
```

3. Start the API:

```powershell
D:\Anaconda\python.exe -m free_llm_router serve --config config\config.yaml
```

4. Start the dashboard in another shell:

```powershell
streamlit run src\free_llm_router\dashboard.py -- --config config\config.yaml
```

The dashboard now includes a `Chat Tester` tab where you can:

- point at the local router `/v1` endpoint
- select a provider or keep all providers enabled
- test a specific model or `auto` routing
- add scenario and performance hints
- inspect the raw router response and selected upstream model

## API usage

### Route automatically

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local-router")

response = client.chat.completions.create(
    model="auto",
    messages=[
        {"role": "system", "content": "You are a precise assistant."},
        {"role": "user", "content": "Reason step by step about the best caching strategy for this API."},
    ],
)

print(response.choices[0].message.content)
```

### Inspect provider setup guidance

```powershell
curl http://127.0.0.1:8000/v1/providers
```

This endpoint exposes the normalized provider metadata used by the wizard and dashboard, including:

- setup reference file
- required environment variables
- auth hints
- resolved base URL
- example model

### Hint the router

You can keep the OpenAI-like payload and add an optional `router` block:

```json
{
  "model": "auto",
  "messages": [{"role": "user", "content": "Write Python code to scrape a sitemap."}],
  "router": {
    "scenario": "coding",
    "performance": "high",
    "minimum_context_tokens": 32000
  }
}
```

## Notes

- API keys are referenced by environment-variable name in config. The wizard never writes raw keys into the YAML file.
- Provider setup metadata is aligned to:
  - `free-llm-apis/references/inference-providers.md`
  - `free-llm-apis/references/provider-apis.md`
- Health checks use lightweight provider-specific probes and persist status in SQLite.
- Streaming pass-through is supported for OpenAI-compatible upstream providers.

## References

- Provider catalog: [mnfst/awesome-free-llm-apis](https://github.com/mnfst/awesome-free-llm-apis)
- Setup guidance: [free-llm-apis/SKILL.md](https://github.com/mnfst/awesome-free-llm-apis/blob/main/free-llm-apis/SKILL.md)
