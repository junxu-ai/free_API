import asyncio
from typing import Optional

from free_llm_router.catalog import Catalog
from free_llm_router.clients import ProviderClient
from free_llm_router.store import RouterStore


class HealthMonitor:
    def __init__(self, catalog: Catalog, store: RouterStore, client: ProviderClient, interval_seconds: int):
        self.catalog = catalog
        self.store = store
        self.client = client
        self.interval_seconds = interval_seconds
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def run_once(self) -> None:
        for model in self.catalog.active_models():
            success, status_code, latency_ms, error = await self.client.healthcheck(model)
            self.store.log_health_check(
                provider_id=model.provider_id,
                model_id=model.id,
                success=success,
                latency_ms=latency_ms,
                status_code=status_code,
                error=error,
            )

    async def _run(self) -> None:
        while self._running:
            await self.run_once()
            await asyncio.sleep(self.interval_seconds)
