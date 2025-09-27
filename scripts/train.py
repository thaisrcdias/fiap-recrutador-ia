
import os
import pandas as pd
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.model_trainer import ModelTrainer

PROJECT_ID = os.getenv("PROJECT_ID", "resolute-spirit-472116-f2")
VIEW_FULL_PATH = os.getenv("VIEW_FULL_PATH", "resolute-spirit-472116-f2.recrutamento.vagas_prospects_applicants_completo")
MODEL_PATH = os.getenv("MODEL_PATH", "saved_models/match_model.joblib")

def main():
    # loader = DataLoader(PROJECT_ID, VIEW_FULL_PATH)
    # df = loader.load_data()
    df = pd.read_csv('../base_gcp.csv', sep=';')
    if df.empty:
        print("Sem dados.")
        return

    if "situacao_candidato" not in df.columns:
        print("Coluna 'situacao_candidato' ausente nos dados.")
        return
    df["target"] = (df["situacao_candidato"] == "Encaminhado ao Requisitante").astype(int)
    print("Distribuição do alvo:", df["target"].value_counts(normalize=True))

    feature_candidates = [
        "titulo_vaga","nivel_vaga","conhecimentos_tecnicos_vaga",
        "cv_pt","area_atuacao","conhecimentos_tecnicos_candidato",
        "nivel_academico_candidato","nivel_ingles_candidato",
        "nivel_espanhol_candidato","tipo_contratacao"
    ]
    use_cols = [c for c in feature_candidates if c in df.columns]
    X_df = df[use_cols]
    y = df["target"]

    pre = Preprocessor()
    X = pre.fit_transform(X_df)

    trainer = ModelTrainer(model_path=MODEL_PATH)
    trainer.train(X, y)
    trainer.save_pipeline(pre)

if __name__ == "__main__":
    os.makedirs("saved_models", exist_ok=True)
    main()