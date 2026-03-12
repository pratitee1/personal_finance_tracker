import asyncio
import importlib
import sys
import types
from unittest.mock import MagicMock

from fastapi import APIRouter
from starlette.responses import Response


def _make_router_module(path: str):
    module = types.ModuleType(path)
    router = APIRouter()

    @router.get("/health")
    async def health():
        return {"ok": True, "path": path}

    module.router = router
    return module


def test_main_registers_routes_and_middleware(monkeypatch):
    routes_package = types.ModuleType("api.routes")
    routes_package.auth = _make_router_module("auth")
    routes_package.upload_receipt = _make_router_module("upload")
    routes_package.rag_qa = _make_router_module("rag")

    monkeypatch.setitem(sys.modules, "api.routes", routes_package)
    monkeypatch.setitem(sys.modules, "api.routes.auth", routes_package.auth)
    monkeypatch.setitem(sys.modules, "api.routes.upload_receipt", routes_package.upload_receipt)
    monkeypatch.setitem(sys.modules, "api.routes.rag_qa", routes_package.rag_qa)
    sys.modules.pop("api.main", None)

    main = importlib.import_module("api.main")
    route_paths = {route.path for route in main.app.router.routes}

    assert "/auth/health" in route_paths
    assert "/upload/health" in route_paths
    assert "/rag/health" in route_paths

    request = MagicMock(method="GET", url="http://testserver/auth/health")

    async def call_next(_request):
        return Response(status_code=204)

    response = asyncio.run(main.log_requests(request, call_next))
    assert response.status_code == 204
