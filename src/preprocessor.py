import re
import pandas as pd
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- Stopwords em português via NLTK ---
import nltk
from nltk.corpus import stopwords

# Baixa stopwords apenas uma vez (primeira execução)
try:
    PT_STOPWORDS = stopwords.words('portuguese')
except LookupError:
    nltk.download('stopwords')
    PT_STOPWORDS = stopwords.words('portuguese')


def _clean_text(txt: str) -> str:
    """Normaliza texto: minúsculo, sem acentos, sem excesso de espaços."""
    txt = unidecode(str(txt).lower())
    txt = re.sub(r'\s+', ' ', txt)  # substitui múltiplos espaços
    return txt.strip()


class Preprocessor:
    """Realiza o pré-processamento e a engenharia de features."""

    def __init__(self):
        # Colunas de entrada que serão transformadas
        self.text_feature_cols = ['texto_vaga', 'texto_candidato']
        self.categorical_feature_cols = [
            'nivel_academico_candidato',
            'nivel_ingles_candidato'
        ]

        # Define o pipeline de pré-processamento
        self.pipeline = ColumnTransformer(
            transformers=[
                (
                    'vaga_tfidf',
                    TfidfVectorizer(stop_words=PT_STOPWORDS, max_features=500),
                    'texto_vaga'
                ),
                (
                    'candidato_tfidf',
                    TfidfVectorizer(stop_words=PT_STOPWORDS, max_features=1000),
                    'texto_candidato'
                ),
                (
                    'categorical',
                    OneHotEncoder(handle_unknown='ignore'),
                    self.categorical_feature_cols
                )
            ],
            remainder='drop'
        )

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria colunas de texto compostas e normaliza valores."""
        dfc = df.copy()
        dfc.fillna("", inplace=True)

        # Concatena atributos para gerar textos mais ricos
        dfc['texto_vaga'] = (
            dfc['titulo_vaga'].astype(str) + " " +
            dfc['nivel_vaga'].astype(str) + " " +
            dfc['conhecimentos_tecnicos_vaga'].astype(str)
        )
        dfc['texto_candidato'] = (
            dfc['cv_pt'].astype(str) + " " +
            dfc['area_atuacao'].astype(str) + " " +
            dfc['conhecimentos_tecnicos_candidato'].astype(str)
        )

        # Limpeza básica dos textos
        for col in ['texto_vaga', 'texto_candidato']:
            dfc[col] = dfc[col].map(_clean_text)

        # Normaliza categorias para letras minúsculas
        for col in self.categorical_feature_cols:
            dfc[col] = dfc[col].astype(str).str.strip().str.lower()

        return dfc

    def fit_transform(self, df: pd.DataFrame):
        """Ajusta e transforma os dados (treino)."""
        prepared = self.prepare_features(df)
        return self.pipeline.fit_transform(prepared)

    def transform(self, df: pd.DataFrame):
        """Transforma os dados com o pipeline já ajustado (inference)."""
        prepared = self.prepare_features(df)
        return self.pipeline.transform(prepared)