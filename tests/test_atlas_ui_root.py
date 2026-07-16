"""Atlas UI-root behavior when the production build is available."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.staticfiles import StaticFiles

from atlas_brain.api import ollama_compat


def test_ui_root_serves_browsers_and_preserves_ollama_health(tmp_path):
    (tmp_path / "index.html").write_text("<h1>Atlas UI</h1>", encoding="utf-8")
    (tmp_path / "app.js").write_text("console.log('Atlas UI')", encoding="utf-8")
    original_ui_dist = ollama_compat._UI_DIST
    ollama_compat._UI_DIST = tmp_path
    try:
        app = FastAPI()
        app.include_router(ollama_compat.router)
        app.mount("/", StaticFiles(directory=str(tmp_path)), name="ui")
        client = TestClient(app)

        browser_response = client.get("/", headers={"accept": "text/html"})
        assert browser_response.status_code == 200
        assert browser_response.text == "<h1>Atlas UI</h1>"
        assert browser_response.headers["content-type"].startswith("text/html")

        ollama_response = client.get("/", headers={"accept": "application/json"})
        assert ollama_response.status_code == 200
        assert ollama_response.text == "Ollama is running"

        asset_response = client.get("/app.js")
        assert asset_response.status_code == 200
        assert asset_response.text == "console.log('Atlas UI')"
    finally:
        ollama_compat._UI_DIST = original_ui_dist
