import numpy as np
from app.predictor import Predictor

def test_predict_with_mock(monkeypatch):
    class MockPreprocessor:
        def transform(self, X):
            return X

    class MockModel:
        def predict(self, X):
            return [1]
        def predict_proba(self, X):
            return np.array([[0.2, 0.8]])

    mock_pipeline = {
        "preprocessor": MockPreprocessor(),
        "model": MockModel()
    }

    monkeypatch.setattr(Predictor, "_load_pipeline", lambda self: mock_pipeline)

    predictor = Predictor()
    result = predictor.predict({
        "titulo_vaga": "Dev",
        "nivel_vaga": "Junior",
        "conhecimentos_tecnicos_vaga": "Python",
        "cv_pt": "Estágio em dados",
        "area_atuacao": "TI",
        "conhecimentos_tecnicos_candidato": "SQL",
        "nivel_academico_candidato": "Graduação",
        "nivel_ingles_candidato": "Básico"
    })

    assert result["prediction"] == 1
    assert 0 <= result["probability"] <= 1
