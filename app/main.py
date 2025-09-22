import types
import pytest
from fastapi.testclient import TestClient
from fastapi.routing import APIRoute
from app.main import app

@pytest.fixture(scope="session")
def client():
    return TestClient(app)

def _app_routes():
    """Retorna apenas rotas de aplicação (exclui OpenAPI/docs)."""
    excluded = {"/openapi.json", "/docs", "/redoc"}
    routes = []
    for r in app.router.routes:
        if isinstance(r, APIRoute) and r.path not in excluded:
            routes.append(r)
    return routes

def test_app_metadata():
    assert app.title == "Recruitment Prediction API"
    assert app.description.startswith("API para predição")
    assert app.version == "1.0.0"

def test_root_ok(client: TestClient):
    res = client.get("/")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body, dict)
    assert "message" in body
    assert "Recruitment API" in body["message"]

def test_openapi_and_docs(client: TestClient):
    res_openapi = client.get("/openapi.json")
    assert res_openapi.status_code == 200
    assert isinstance(res_openapi.json(), dict)

    res_docs = client.get("/docs")
    assert 200 <= res_docs.status_code < 400

    res_redoc = client.get("/redoc")
    assert 200 <= res_redoc.status_code < 400

@pytest.mark.parametrize("path", ["/health", "/healthz", "/livez", "/readyz"])
def test_optional_health_endpoints(client: TestClient, path: str):
    """Se endpoints de health não existirem, o teste é ignorado."""
    res = client.get(path)
    if res.status_code == 404:
        pytest.skip(f"Endpoint opcional {path} não existe (ok).")
    assert res.status_code == 200

def test_has_at_least_one_app_route():
    """Garante ao menos 1 rota de aplicação (o `/` já atende)."""
    routes = _app_routes()
    assert len(routes) >= 1, "Nenhuma rota de aplicação encontrada."

def test_routes_are_callable():
    """Handlers de rotas devem ser chamáveis (evita import quebrado)."""
    for r in _app_routes():
        handler = r.endpoint
        assert callable(handler), f"Handler não chamável para rota {r.path}"
        assert isinstance(handler, (types.FunctionType, types.MethodType)), \
            f"Endpoint inesperado para {r.path}"
