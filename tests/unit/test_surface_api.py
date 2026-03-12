from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from aurora.host_runtime.config import AuroraSettings
from aurora.host_runtime.errors import CollapseProviderError
from aurora.host_runtime.provider import CollapseProvider
from aurora.host_runtime.runtime import AuroraRuntime
from aurora.surface_api.app import build_app
from tests.helpers.fakes import FakeCollapseProvider, FakeSubstrateClient


def _runtime_factory(tmp_path: Path):
    def factory() -> AuroraRuntime:
        return AuroraRuntime(
            settings=AuroraSettings(data_dir=str(tmp_path)),
            substrate_client=FakeSubstrateClient(),
            provider=FakeCollapseProvider(),
        )

    return factory


class _FailingProvider(CollapseProvider):
    def collapse(self, request):
        raise CollapseProviderError("provider unavailable")

    def is_healthy(self) -> bool:
        return False

    def ensure_ready(self) -> None:
        return None

    def provider_identity(self) -> str:
        return "failing"


def test_surface_api_exposes_only_seed_v1_routes(tmp_path: Path) -> None:
    app = build_app(runtime_factory=_runtime_factory(tmp_path))
    client = TestClient(app)

    response = client.post("/v1/input", json={"user_text": "hello", "language": "en"})
    assert response.status_code == 200
    assert "event_id" in response.json()

    alias_response = client.post("/v1/input", json={"text": "hello again", "language": "en"})
    assert alias_response.status_code == 200
    assert "event_id" in alias_response.json()

    health = client.get("/v1/healthz")
    assert health.status_code == 200
    assert "anchor_count" in health.json()
    assert "substrate_alive" in health.json()

    integrity = client.get("/v1/integrity")
    assert integrity.status_code == 200
    assert "runtime_boundary" in integrity.json()

    routes = {route.path for route in app.routes}
    assert routes == {"/v1/input", "/v1/healthz", "/v1/integrity"}


def test_surface_api_maps_provider_failures_to_502(tmp_path: Path) -> None:
    app = build_app(
        runtime_factory=lambda: AuroraRuntime(
            settings=AuroraSettings(data_dir=str(tmp_path)),
            substrate_client=FakeSubstrateClient(),
            provider=_FailingProvider(),
        )
    )
    client = TestClient(app)

    response = client.post("/v1/input", json={"text": "hello", "language": "en"})

    assert response.status_code == 502
    assert response.json()["detail"] == "provider unavailable"
