from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()

def test_predict_endpoint(monkeypatch):
    def mock_predict(data):
        return {"prediction": 1, "probability": 0.9, "message": "Mocked"}
    from app import predictor
    monkeypatch.setattr(predictor.Predictor, "predict", mock_predict)

    response = client.post("/predict", json={
        "titulo_vaga": "Engenheiro",
        "nivel_vaga": "Senior",
        "conhecimentos_tecnicos_vaga": "Python",
        "cv_pt": "5 anos de experiência",
        "area_atuacao": "Dados",
        "conhecimentos_tecnicos_candidato": "SQL",
        "nivel_academico_candidato": "Mestrado",
        "nivel_ingles_candidato": "avançado"
    })
    assert response.status_code == 200
    assert response.json()["prediction"] == 1
