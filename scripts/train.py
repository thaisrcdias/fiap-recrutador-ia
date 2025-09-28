
# import os
# import pandas as pd
# from src.data_loader import DataLoader
# from src.preprocessor import Preprocessor
# from src.model_trainer import ModelTrainer

# PROJECT_ID = os.getenv("PROJECT_ID", "resolute-spirit-472116-f2")
# VIEW_FULL_PATH = os.getenv("VIEW_FULL_PATH", "resolute-spirit-472116-f2.recrutamento.vagas_prospects_applicants_completo")
# MODEL_PATH = os.getenv("MODEL_PATH", "saved_models/match_model.joblib")

# def main():
#     loader = DataLoader(PROJECT_ID, VIEW_FULL_PATH)
#     df = loader.load_data()
#     if df.empty:
#         print("Sem dados.")
#         return

#     if "situacao_candidato" not in df.columns:
#         print("Coluna 'situacao_candidato' ausente nos dados.")
#         return
#     df["target"] = (df["situacao_candidato"] == "Encaminhado ao Requisitante").astype(int)
#     print("Distribuição do alvo:", df["target"].value_counts(normalize=True))

#     feature_candidates = [
#         "titulo_vaga","nivel_vaga","conhecimentos_tecnicos_vaga",
#         "cv_pt","area_atuacao","conhecimentos_tecnicos_candidato",
#         "nivel_academico_candidato","nivel_ingles_candidato",
#         "nivel_espanhol_candidato","tipo_contratacao"
#     ]
#     use_cols = [c for c in feature_candidates if c in df.columns]
#     X_df = df[use_cols]
#     y = df["target"]

#     pre = Preprocessor()
#     X = pre.fit_transform(X_df)

#     trainer = ModelTrainer(model_path=MODEL_PATH)
#     trainer.train(X, y)
#     trainer.save_pipeline(pre)

# if __name__ == "__main__":
#     os.makedirs("saved_models", exist_ok=True)
#     main()


# import os, tempfile
# import pandas as pd
# from src.data_loader import DataLoader
# from src.preprocessor import Preprocessor
# from src.model_trainer import ModelTrainer
# from google.cloud import storage

# PROJECT_ID = os.getenv("PROJECT_ID", "resolute-spirit-472116-f2")
# VIEW_FULL_PATH = os.getenv("VIEW_FULL_PATH", "resolute-spirit-472116-f2.recrutamento.vagas_prospects_applicants_completo")
# MODEL_PATH = os.getenv("MODEL_PATH", "gs://resolute-spirit-472116-f2-mlops-artifacts/models/recrutador-match/v1/match_model.joblib")


# def download_gcs_to_local(gs_uri: str) -> str:
#     # gs_uri: gs://bucket/path/file.joblib
#     assert gs_uri.startswith("gs://")
#     _, _, bucket, *path = gs_uri.split("/")
#     blob_path = "/".join(path)

#     client = storage.Client()
#     bucket = client.bucket(bucket)
#     blob = bucket.blob(blob_path)

#     fd, local_path = tempfile.mkstemp(suffix=os.path.splitext(blob_path)[-1])
#     os.close(fd)
#     blob.download_to_filename(local_path)
#     return local_path


# def main():
#     loader = DataLoader(PROJECT_ID, VIEW_FULL_PATH)
#     df = loader.load_data()
#     if df.empty:
#         print("Sem dados.")
#         return

#     if "situacao_candidato" not in df.columns:
#         print("Coluna 'situacao_candidato' ausente nos dados.")
#         return
#     df["target"] = (df["situacao_candidato"] == "Encaminhado ao Requisitante").astype(int)
#     print("Distribuição do alvo:", df["target"].value_counts(normalize=True))

#     feature_candidates = [
#         "titulo_vaga","nivel_vaga","conhecimentos_tecnicos_vaga",
#         "cv_pt","area_atuacao","conhecimentos_tecnicos_candidato",
#         "nivel_academico_candidato","nivel_ingles_candidato",
#         "nivel_espanhol_candidato","tipo_contratacao"
#     ]
#     use_cols = [c for c in feature_candidates if c in df.columns]
#     X_df = df[use_cols]
#     y = df["target"]

#     pre = Preprocessor()
#     X = pre.fit_transform(X_df)

#     trainer = ModelTrainer(model_path=MODEL_PATH)
#     trainer.train(X, y)
#     trainer.save_pipeline(pre)

# if __name__ == "__main__":
#     os.makedirs("saved_models", exist_ok=True)
#     main()

# scripts/train_or_load.py
# from __future__ import annotations

# """
# Treina OU carrega bundle (preprocessor + model) a partir de um gs://... em GCS.

# - Se MODEL_PATH existir no GCS e RETRAIN != 1:
#     -> baixa o artefato e carrega o bundle para uso.
# - Caso contrário:
#     -> treina com DataLoader + Preprocessor + ModelTrainer
#     -> salva bundle local temporário e faz upload para o GCS (MODEL_PATH).

# Variáveis de ambiente úteis:
#   PROJECT_ID=resolute-spirit-472116-f2
#   VIEW_FULL_PATH=resolute-spirit-472116-f2.recrutamento.vagas_prospects_applicants_completo
#   MODEL_PATH=gs://<bucket>/models/recrutador-match/v1/match_model.joblib
#   RETRAIN=0|1
#   LOG_LEVEL=INFO|DEBUG

# Requisitos:
#   pip install google-cloud-storage joblib pandas scikit-learn
#   (e suas libs do seu projeto já existentes)

# Credenciais:
#   GOOGLE_APPLICATION_CREDENTIALS=<path/service-account.json>
# """

# import os
# import tempfile
# import logging
# from typing import Tuple, Dict, Any

# import joblib
# import numpy as np
# import pandas as pd
# from google.cloud import storage

# from src.data_loader import DataLoader
# from src.preprocessor import Preprocessor
# from src.model_trainer import ModelTrainer

# # ------------------ Config ------------------
# PROJECT_ID = os.getenv("PROJECT_ID", "resolute-spirit-472116-f2")
# VIEW_FULL_PATH = os.getenv(
#     "VIEW_FULL_PATH",
#     "resolute-spirit-472116-f2.recrutamento.vagas_prospects_applicants_completo",
# )
# MODEL_PATH = os.getenv(
#     "MODEL_PATH",
#     "gs://resolute-spirit-472116-f2-mlops-artifacts/models/recrutador-match/v1/match_model.joblib",
# )
# print(MODEL_PATH)
# RETRAIN = os.getenv("RETRAIN", "1").lower() in {"1", "true", "yes"}
# LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
# logger = logging.getLogger("train_or_load")


# # ------------------ Utils GCS ------------------
# def _parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
#     if not gs_uri.startswith("gs://"):
#         raise ValueError(f"Esperado gs://… em MODEL_PATH. Recebido: {gs_uri}")
#     _, _, bucket, *path = gs_uri.split("/")
#     return bucket, "/".join(path)


# def gcs_exists(gs_uri: str) -> bool:
#     bucket_name, blob_path = _parse_gs_uri(gs_uri)
#     client = storage.Client()
#     blob = client.bucket(bucket_name).blob(blob_path)
#     return blob.exists()


# def gcs_download_to_temp(gs_uri: str) -> str:
#     """Baixa o gs://… para um arquivo temporário e retorna o caminho local."""
#     bucket_name, blob_path = _parse_gs_uri(gs_uri)
#     client = storage.Client()
#     blob = client.bucket(bucket_name).blob(blob_path)

#     suffix = os.path.splitext(blob_path)[-1] or ".joblib"
#     fd, local_path = tempfile.mkstemp(suffix=suffix)
#     os.close(fd)
#     logger.info("Baixando artefato do GCS: %s -> %s", gs_uri, local_path)
#     blob.download_to_filename(local_path)
#     return local_path


# def gcs_upload(local_path: str, gs_uri: str) -> None:
#     bucket_name, blob_path = _parse_gs_uri(gs_uri)
#     client = storage.Client()
#     blob = client.bucket(bucket_name).blob(blob_path)
#     logger.info("Enviando artefato ao GCS: %s -> %s", local_path, gs_uri)
#     blob.upload_from_filename(local_path)


# def load_bundle_from_gcs(gs_uri: str) -> Dict[str, Any]:
#     """Carrega {'preprocessor': ..., 'model': ...} de um gs://… (via download para tmp)."""
#     local_path = gcs_download_to_temp(gs_uri)
#     try:
#         logger.info("Carregando bundle joblib: %s", local_path)
#         bundle = joblib.load(local_path)
#     finally:
#         # opcional: remover o arquivo temporário
#         try:
#             os.remove(local_path)
#         except Exception:
#             pass
#     return bundle


# # ------------------ Treino + Upload ------------------
# def train_and_push(df: pd.DataFrame, model_path_gs: str) -> Dict[str, Any]:
#     """Treina o pipeline e publica no GCS como bundle joblib."""
#     if "situacao_candidato" not in df.columns:
#         raise RuntimeError("Coluna 'situacao_candidato' ausente — não há rótulo para treino.")

#     # Exemplo simples de target (ajuste sua regra aqui se precisar)
#     df = df.copy()
#     df["target"] = (df["situacao_candidato"].astype(str).str.lower() ==
#                     "encaminhado ao requisitante".lower()).astype(int)
#     logger.info("Distribuição do alvo (1/0):\n%s", df["target"].value_counts(dropna=False))

#     feature_candidates = [
#         "titulo_vaga", "nivel_vaga", "conhecimentos_tecnicos_vaga",
#         "cv_pt", "area_atuacao", "conhecimentos_tecnicos_candidato",
#         "nivel_academico_candidato", "nivel_ingles_candidato",
#         "nivel_espanhol_candidato", "tipo_contratacao",
#     ]
#     use_cols = [c for c in feature_candidates if c in df.columns]
#     if not use_cols:
#         raise RuntimeError("Nenhuma feature disponível para treino.")
#     X_df = df[use_cols].fillna("")
#     y = df["target"].values

#     pre = Preprocessor()
#     X = pre.fit_transform(X_df)
#     logger.info("Shape pós-preprocessamento: %s", getattr(X, "shape", None))

#     trainer = ModelTrainer(model_path=model_path_gs)
#     trainer.train(X, y)

#     # criamos um bundle com preprocessor + modelo (padrão do seu projeto)
#     fd, local_bundle = tempfile.mkstemp(suffix=".joblib")
#     os.close(fd)
#     joblib.dump({"preprocessor": pre, "model": trainer.model}, local_bundle)
#     logger.info("Bundle salvo localmente: %s", local_bundle)

#     # publica no GCS
#     gcs_upload(local_bundle, model_path_gs)

#     # cleanup
#     try:
#         os.remove(local_bundle)
#     except Exception:
#         pass

#     return {"preprocessor": pre, "model": trainer.model}


# # ------------------ Main ------------------
# def main() -> None:
#     global MODEL_PATH, RETRAIN
#     logger.info("Carregando dados da VIEW: %s", VIEW_FULL_PATH)
#     loader = DataLoader(PROJECT_ID, VIEW_FULL_PATH)
#     df = loader.load_data()
#     if df.empty:
#         print("Sem dados.")
#         return

#     logger.info("MODEL_PATH=%s | RETRAIN=%s", MODEL_PATH, RETRAIN)

#     if not RETRAIN and gcs_exists(MODEL_PATH):
#         logger.info("Artefato encontrado no GCS. Carregando bundle existente.")
#         bundle = load_bundle_from_gcs(MODEL_PATH)
#     else:
#         logger.info("Treinando (forçado ou artefato ausente).")
#         MODEL_PATH = MODEL_PATH.replace("v1", "v2")  # para capturar no closure
#         bundle = train_and_push(df, MODEL_PATH)

#     # Sanity-check: usar o bundle carregado/treinado
#     pre = bundle["preprocessor"]
#     model = bundle["model"]

#     feature_candidates = [
#         "titulo_vaga", "nivel_vaga", "conhecimentos_tecnicos_vaga",
#         "cv_pt", "area_atuacao", "conhecimentos_tecnicos_candidato",
#         "nivel_academico_candidato", "nivel_ingles_candidato",
#         "nivel_espanhol_candidato", "tipo_contratacao",
#     ]
#     use_cols = [c for c in feature_candidates if c in df.columns]
#     X_sample = pre.transform(df[use_cols].fillna("").head(50))
#     proba = model.predict_proba(X_sample)[:, 1]
#     logger.info("Sanity-check: média das probabilidades (50 amostras): %.4f", float(np.mean(proba)))


# if __name__ == "__main__":
#     main()

# scripts/train.py
# """
# Treina o modelo clássico (scikit-learn) de matching de candidatos vs. vaga.

# Fluxo:
#   1) Carrega dados da VIEW do BigQuery (via src.data_loader.DataLoader) ou de um CSV.
#   2) Constrói o alvo (y) a partir de 'situacao_candidato' (configurável).
#   3) Aplica src.preprocessor.Preprocessor e treina um classificador (ModelTrainer).
#   4) Salva o bundle {'preprocessor', 'model'} localmente ou no GCS (se MODEL_PATH for 'gs://...').

# Requisitos:
#   - GOOGLE_APPLICATION_CREDENTIALS (para BQ/GCS)
#   - pip installs já citados no projeto (google-cloud-bigquery, google-cloud-storage, scikit-learn, joblib etc.)

# Execução (da raiz do projeto):
#   python -m scripts.train --retrain
#   python -m scripts.train --csv data/export.csv --retrain
#   python -m scripts.train --retrain --model-path gs://<bucket>/models/recrutador-match/v1/match_model.joblib
# """

# from __future__ import annotations

# import argparse
# import logging
# import os
# from typing import Tuple, Iterable

# import numpy as np
# import pandas as pd

# # Nossos módulos
# from src.data_loader import DataLoader
# from src.preprocessor import Preprocessor
# from src.model_trainer import ModelTrainer

# # -------------------------
# # Logging básico
# # -------------------------
# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     level=os.getenv("LOG_LEVEL", "INFO"),
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
# )

# # -------------------------
# # Config via ambiente (globais só para default)
# # -------------------------
# PROJECT_ID = os.getenv("PROJECT_ID", "resolute-spirit-472116-f2")
# VIEW_FULL_PATH = os.getenv(
#     "VIEW_FULL_PATH",
#     "resolute-spirit-472116-f2.recrutamento.vagas_prospects_applicants_completo",
# )
# MODEL_PATH = os.getenv(
#     "MODEL_PATH",
#     "gs://resolute-spirit-472116-f2-mlops-artifacts/models/recrutador-match/v1/match_model.joblib",
# )
# RETRAIN = os.getenv("RETRAIN", "false").lower() in {"1", "true", "yes", "y"}

# # Colunas de features candidatas (serão filtradas pelas que existem no DF)
# FEATURE_CANDIDATES: Tuple[str, ...] = (
#     "titulo_vaga",
#     "nivel_vaga",
#     "conhecimentos_tecnicos_vaga",
#     "cv_pt",
#     "area_atuacao",
#     "conhecimentos_tecnicos_candidato",
#     "nivel_academico_candidato",
#     "nivel_ingles_candidato",
#     "nivel_espanhol_candidato",
#     "tipo_contratacao",
# )

# # Situações consideradas "positivas" no histórico (ajuste conforme sua realidade)
# POSITIVE_STATUSES: Tuple[str, ...] = (
#     "encaminhado ao requisitante",
#     "contratado pela decision",
#     "contratado",
# )


# # -------------------------
# # Helpers
# # -------------------------
# def parse_args() -> argparse.Namespace:
#     """Parse de argumentos de linha de comando."""
#     ap = argparse.ArgumentParser(description="Treino do modelo de matching (scikit-learn).")
#     ap.add_argument("--project-id", default=None, help="ID do projeto GCP (override).")
#     ap.add_argument(
#         "--view",
#         default=None,
#         help="Caminho completo da VIEW do BigQuery (ex.: <project>.<dataset>.<view>)",
#     )
#     ap.add_argument(
#         "--model-path",
#         default=None,
#         help="Destino do bundle (local ou GCS 'gs://bucket/path/match_model.joblib').",
#     )
#     ap.add_argument(
#         "--retrain",
#         action="store_true",
#         help="Força retreino do modelo (ignora bundle existente).",
#     )
#     ap.add_argument("--csv", default=None, help="Caminho CSV para treinar sem BigQuery.")
#     return ap.parse_args()


# def load_dataframe(project_id: str, view_full_path: str, csv_path: str | None) -> pd.DataFrame:
#     """
#     Carrega dados para treino.

#     - Se `csv_path` for passado, lê do CSV.
#     - Caso contrário, usa DataLoader (BigQuery).

#     Raises:
#         RuntimeError: se o DataFrame vier vazio.
#     """
#     if csv_path:
#         logger.info("Carregando dados a partir de CSV: %s", csv_path)
#         df = pd.read_csv(csv_path)
#     else:
#         logger.info("Carregando dados da VIEW: %s", view_full_path)
#         loader = DataLoader(project_id, view_full_path)
#         try:
#             df = loader.load_data()
#         except Exception as e:
#             logger.error("Erro ao carregar dados do BigQuery: %s", e)
#             raise

#     if df is None or df.empty:
#         raise RuntimeError("Dataset vazio.")
#     logger.info("Dados carregados: %d linhas / %d colunas", len(df), len(df.columns))
#     return df


# def build_target(df: pd.DataFrame, pos_statuses: Iterable[str] = POSITIVE_STATUSES) -> pd.Series:
#     """
#     Constrói o alvo binário a partir de 'situacao_candidato'.
#     1 para status "positivo", 0 caso contrário.
#     """
#     if "situacao_candidato" not in df.columns:
#         raise RuntimeError("Coluna 'situacao_candidato' não encontrada.")

#     status = df["situacao_candidato"].astype(str).str.strip().str.lower()
#     positive = status.isin(set(s.lower() for s in pos_statuses))
#     y = positive.astype(int)

#     # Log de distribuição
#     vals, cnts = np.unique(y.values, return_counts=True)
#     dist = {int(k): int(v) for k, v in zip(vals, cnts)}
#     logger.info("Distribuição do alvo (0/1): %s", dist)
#     return y


# def select_feature_columns(df: pd.DataFrame) -> list[str]:
#     """Seleciona as colunas de features que estão presentes no DF."""
#     cols = [c for c in FEATURE_CANDIDATES if c in df.columns]
#     if not cols:
#         raise RuntimeError(
#             f"Nenhuma coluna de feature encontrada. Esperava uma entre: {FEATURE_CANDIDATES}"
#         )
#     logger.info("Features selecionadas (%d): %s", len(cols), cols)
#     return cols


# def train_and_push(df: pd.DataFrame, model_path: str) -> str:
#     """
#     Treina o modelo com Preprocessor e salva o bundle no destino indicado.

#     Returns:
#         Caminho de destino (local ou gs://) do bundle salvo.
#     """
#     # 1) Alvo
#     y = build_target(df)

#     # 2) Features
#     feat_cols = select_feature_columns(df)
#     X_df = df[feat_cols]

#     # 3) Preprocessamento
#     pre = Preprocessor()
#     X = pre.fit_transform(X_df)

#     # 4) Treino e salvamento
#     trainer = ModelTrainer(model_path=model_path)
#     trainer.train(X, y)
#     saved_path = trainer.save_pipeline(pre)

#     logger.info("Bundle salvo em: %s", saved_path)
#     return saved_path


# # -------------------------
# # Main
# # -------------------------
# def main() -> None:
#     """Ponto de entrada do script de treino."""
#     args = parse_args()

#     project_id = args.project_id or PROJECT_ID
#     view_path = args.view or VIEW_FULL_PATH
#     model_path = args.model_path or MODEL_PATH
#     retrain = bool(args.retrain or RETRAIN)
#     csv_path = args.csv

#     logger.info("PROJECT_ID=%s", project_id)
#     logger.info("VIEW_FULL_PATH=%s", view_path)
#     logger.info("MODEL_PATH=%s | RETRAIN=%s", model_path, retrain)

#     # Carrega dados
#     df = load_dataframe(project_id, view_path, csv_path)

#     # (Opcional) Se não quiser sempre retreinar, aqui daria para tentar carregar o bundle existente
#     # e pular o treino se quiser. Para manter o requisito da disciplina, vamos treinar quando
#     # --retrain estiver ligado (ou sempre treinar nesta entrega).
#     if not retrain:
#         logger.warning(
#             "Flag --retrain não informada. Por requisitos de entrega, seguiremos treinando mesmo assim."
#         )

#     # Executa treino + persistência
#     try:
#         saved = train_and_push(df, model_path)
#         logger.info("Treino concluído e bundle salvo em: %s", saved)
#     except Exception as e:
#         logger.exception("Falha no treino: %s", e)
#         raise


# if __name__ == "__main__":
#     main()

# scripts/train.py
from __future__ import annotations

import logging
import os
from typing import List

import numpy as np
import pandas as pd

# Importa seus módulos do projeto
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.model_trainer import ModelTrainer

# -----------------------------
# Configuração básica de logging
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------
# Variáveis de ambiente
# -----------------------------
PROJECT_ID = os.getenv("PROJECT_ID", "resolute-spirit-472116-f2")
VIEW_FULL_PATH = os.getenv(
    "VIEW_FULL_PATH",
    "resolute-spirit-472116-f2.recrutamento.vagas_prospects_applicants_completo",
)
# Se começar com gs://, o ModelTrainer fará upload para o GCS
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "gs://resolute-spirit-472116-f2-mlops-artifacts/models/recrutador-match/v1/match_model.joblib",
)

# Colunas candidatas de features (use as que existirem no schema)
FEATURE_CANDIDATES: List[str] = [
    "titulo_vaga",
    "nivel_vaga",
    "conhecimentos_tecnicos_vaga",
    "cv_pt",
    "area_atuacao",
    "conhecimentos_tecnicos_candidato",
    "nivel_academico_candidato",
    "nivel_ingles_candidato",
    "nivel_espanhol_candidato",
    "tipo_contratacao",
]

TARGET_COL = "situacao_candidato"   # origem do rótulo
POSITIVE_VALUE = "Encaminhado ao Requisitante"  # define classe positiva


def load_training_data(project_id: str, view_path: str) -> pd.DataFrame:
    """Carrega dados da VIEW via DataLoader."""
    logger.info("Carregando dados da VIEW: %s", view_path)
    loader = DataLoader(project_id, view_path)
    df = loader.load_data()
    if df is None or df.empty:
        raise RuntimeError("DataFrame vazio ao carregar a VIEW.")
    logger.info("Dados carregados: %d linhas, %d colunas", len(df), len(df.columns))
    return df


def build_target(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.Series:
    """Cria vetor y binário a partir da coluna de situação/estado."""
    if target_col not in df.columns:
        raise ValueError(f"Coluna '{target_col}' ausente no dataset.")
    y = (df[target_col].astype(str) == POSITIVE_VALUE).astype(int)
    pos_rate = y.mean() if len(y) else 0.0
    logger.info("Distribuição do alvo: pos=%.3f (1), neg=%.3f (0)", pos_rate, 1 - pos_rate)
    return y


def select_feature_frame(df: pd.DataFrame, candidates: List[str]) -> pd.DataFrame:
    """Seleciona apenas as colunas de features que existem no DataFrame."""
    use_cols = [c for c in candidates if c in df.columns]
    if not use_cols:
        raise ValueError(
            f"Nenhuma coluna de features encontrada. "
            f"Candidatas: {candidates} | Colunas: {df.columns.tolist()}"
        )
    X_df = df[use_cols].copy()
    X_df.fillna("", inplace=True)
    logger.info("Features selecionadas: %s", use_cols)
    return X_df


def train_and_push(df: pd.DataFrame, model_path: str) -> str:
    """
    Treina o pipeline (Preprocessor + LogisticRegression) e salva o bundle.
    Se model_path começar com gs://, o upload para GCS é feito pelo ModelTrainer.
    """
    # 1) Target
    y = build_target(df, TARGET_COL)
    # Sanitiza: precisa ter 2 classes distintas
    uniq = np.unique(y.values)
    if uniq.size < 2:
        raise ValueError(
            f"Alvo com {uniq.size} classe(m) {uniq.tolist()}. "
            f"É necessário ter ao menos 2 classes (0 e 1) para treinar."
        )

    # 2) Features
    X_df = select_feature_frame(df, FEATURE_CANDIDATES)

    # 3) Preprocessamento
    pre = Preprocessor()
    logger.info("Ajustando preprocessor (fit_transform)...")
    X = pre.fit_transform(X_df)
    logger.info("Shape pós-preprocessamento: %s", (getattr(X, 'shape', None)))

    # 4) Treino + Persistência
    trainer = ModelTrainer(model_path=model_path)
    trainer.train(X, y)
    saved_uri = trainer.save_pipeline(pre)  # salva local ou GCS conforme model_path
    logger.info("Bundle salvo em: %s", saved_uri)
    return saved_uri


def main() -> None:
    logger.info("Iniciando treino | PROJECT_ID=%s", PROJECT_ID)
    logger.info("VIEW_FULL_PATH=%s", VIEW_FULL_PATH)
    logger.info("MODEL_PATH=%s", MODEL_PATH)

    try:
        df = load_training_data(PROJECT_ID, VIEW_FULL_PATH)
        saved = train_and_push(df, MODEL_PATH)
        logger.info("Treino OK. Artefato publicado em: %s", saved)
    except Exception as e:
        logger.exception("Falha no pipeline de treino: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
