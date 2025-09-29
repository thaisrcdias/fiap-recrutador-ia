import pandas as pd
from src.preprocessor import Preprocessor

def test_preprocessor_fit_transform():
    df = pd.DataFrame({
        "titulo_vaga": ["Engenheiro"],
        "nivel_vaga": ["Senior"],
        "conhecimentos_tecnicos_vaga": ["Python"],
        "cv_pt": ["5 anos de experiência"],
        "area_atuacao": ["Dados"],
        "conhecimentos_tecnicos_candidato": ["SQL"],
        "nivel_academico_candidato": ["Mestrado"],
        "nivel_ingles_candidato": ["avançado"]
    })
    pre = Preprocessor()
    X = pre.fit_transform(df)
    assert X.shape[0] == 1
