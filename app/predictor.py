import os
import joblib
import pandas as pd

class Predictor:
    """Classe responsável por carregar o pipeline e realizar previsões."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.getenv(
            "MODEL_PATH", "app/saved_models/match_model.joblib"
        )
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        """Carrega o pipeline salvo (pré-processador + modelo)."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {self.model_path}")
        return joblib.load(self.model_path)

    def predict(self, input_data: dict) -> dict:
        """
        Realiza predição para um único candidato.
        Args:
            input_data (dict): dados de entrada no mesmo formato das colunas de treino.
        Returns:
            dict: resultado com classe prevista e probabilidade.
        """
        df = pd.DataFrame([input_data])  # converte JSON -> DataFrame

        preprocessor = self.pipeline["preprocessor"]
        model = self.pipeline["model"]

        # transforma entrada
        X = preprocessor.transform(df)

        # gera predição
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0, 1]

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "message": (
                "Candidato com alta probabilidade de ser encaminhado."
                if prediction == 1
                else "Candidato com baixa probabilidade de ser encaminhado."
            ),
        }
