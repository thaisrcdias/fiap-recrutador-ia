import pandas as pd
from src.model_trainer import ModelTrainer
from src.preprocessor import Preprocessor

def test_train_and_save(tmp_path):
    df = pd.DataFrame({
        "titulo_vaga": ["Dev"],
        "nivel_vaga": ["Junior"],
        "conhecimentos_tecnicos_vaga": ["SQL"],
        "cv_pt": ["estágio em dados"],
        "area_atuacao": ["TI"],
        "conhecimentos_tecnicos_candidato": ["SQL"],
        "nivel_academico_candidato": ["graduação"],
        "nivel_ingles_candidato": ["básico"],
        "target": [1]
    })
    X = Preprocessor().fit_transform(df)
    y = df["target"]

    model_path = tmp_path / "test_model.joblib"
    trainer = ModelTrainer(model_path=str(model_path))
    trainer.train(X, y)
    trainer.save_pipeline(Preprocessor())

    assert model_path.exists()
