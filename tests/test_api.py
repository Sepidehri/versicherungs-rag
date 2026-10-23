import pytest
from fastapi.testclient import TestClient

from versicherungs_rag.api import create_app
from versicherungs_rag.components import Components
from versicherungs_rag.config import RetrievalMode, Settings
from versicherungs_rag.llm import ModelRegistry, ModelSpec
from versicherungs_rag.retrieval import HashingEmbedder


@pytest.fixture
def client(index):
    settings = Settings(embedding_model="hashing", retrieval_mode=RetrievalMode.HYBRID, default_model="baseline")
    comps = Components(settings)
    # Inject pre-built parts instead of loading model weights from disk.
    comps.__dict__["index"] = index
    comps.__dict__["embedder"] = HashingEmbedder()
    comps.__dict__["models"] = ModelRegistry(
        [
            ModelSpec(name="baseline", provider="extractive"),
            ModelSpec(name="offline-ollama", provider="ollama", model="none:1b", base_url="http://127.0.0.1:9"),
        ]
    )
    with TestClient(create_app(comps)) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] > 30


def test_ask_returns_answer_with_sources(client):
    response = client.post("/ask", json={"question": "Wie hoch ist der Selbstbehalt in der Vollkasko?"})
    assert response.status_code == 200
    body = response.json()
    assert "350" in body["answer"]
    assert body["contexts"] and body["citations"]


def test_unavailable_model_gives_503(client):
    response = client.post("/ask", json={"question": "Selbstbehalt?", "model": "offline-ollama"})
    assert response.status_code == 503


def test_unknown_model_gives_404(client):
    assert client.post("/ask", json={"question": "Selbstbehalt?", "model": "nope"}).status_code == 404


def test_compare_reports_errors_per_model(client):
    response = client.post(
        "/compare", json={"question": "Selbstbehalt Teilkasko?", "models": ["baseline", "offline-ollama"]}
    )
    results = response.json()
    assert "answer" in results[0]
    assert "error" in results[1]


def test_models_lists_availability(client):
    models = {m["name"]: m for m in client.get("/models").json()}
    assert models["baseline"]["available"] is True
    assert models["offline-ollama"]["available"] is False
