from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List

import joblib
import numpy as np
import pandas as pd
from .llm.config import Config

logger = logging.getLogger(__name__)

# Lista de skills simples para extrair do texto do CV (ilustrativo)
SKILLS = [
    "python", "java", "sql", "bigquery", "spark",
    "aws", "gcp", "sap", "oracle", "linux",
    "docker", "kubernetes", "airflow", "etl", "devops",
]


def _ensure_feature_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Garante que as colunas necessárias ao preprocessor existam no DataFrame.

    Args:
        df: DataFrame de entrada (view unificada).
        cfg: Config com nomes de colunas esperadas pelo preprocessor/modelo.

    Returns:
        Um novo DataFrame com as colunas faltantes adicionadas (strings vazias).
    """
    needed = {
        cfg.col_titulo_vaga,
        cfg.col_nivel_vaga,
        cfg.col_conh_vaga,
        cfg.col_cv_pt,
        cfg.col_area_atuacao,
        cfg.col_conh_cand,
    }
    for c in needed:
        if c not in df.columns:
            logger.debug("Coluna '%s' ausente — criando vazia para inferência.", c)
            df[c] = ""
    return df


def _skills_from_text(text: str) -> List[str]:
    """Extrai uma lista de skills que aparecem no texto (match literal, lowercase).

    Args:
        text: Texto do currículo/campo de conhecimentos.

    Returns:
        Lista de skills presentes no texto.
    """
    t = (text or "").lower()
    return [s for s in SKILLS if s in t]


@lru_cache(maxsize=4)
def _load_bundle(model_path: str) -> dict:
    """Carrega e cacheia o bundle {preprocessor, model} salvo via joblib.

    O cache evita re-load em chamadas repetidas. Para invalidar, troque o path
    ou reinicie o processo (ou mude o maxsize, se necessário).

    Args:
        model_path: Caminho do arquivo .joblib serializado.

    Returns:
        Dicionário com chaves 'preprocessor' e 'model'.

    Raises:
        FileNotFoundError: Se o caminho do modelo não existir.
        RuntimeError: Se o bundle não tiver as chaves esperadas.
    """
    if not os.path.exists(model_path):
        logger.error("Modelo não encontrado em: %s", model_path)
        raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")
    logger.info("Carregando bundle de modelo: %s", model_path)
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "preprocessor" not in bundle or "model" not in bundle:
        logger.error("Bundle inválido: chaves esperadas {'preprocessor','model'} ausentes.")
        raise RuntimeError("Bundle inválido: chaves esperadas {'preprocessor','model'} ausentes.")
    return bundle


def rank_top_k_for_job(
    df_view: pd.DataFrame,
    cfg: Config,
    job_id: str,
    top_k: int = 10,
) -> pd.DataFrame:
    """Ranqueia os TOP-N candidatos para uma vaga.

    Pipeline:
      1) Carrega o bundle (preprocessor + model) do caminho em `cfg.model_path`.
      2) Filtra a view para linhas daquela `job_id`.
      3) Garante colunas usadas pelo preprocessor.
      4) Transforma e pontua (probabilidade da classe positiva).
      5) Agrega por candidato (média se houver duplicatas) e ordena desc.

    Args:
        df_view: DataFrame com registros (por exemplo, a VIEW do BigQuery já carregada).
        cfg: Objeto de configuração contendo nomes de colunas, caminho do modelo, etc.
        job_id: Identificador da vaga (string ou numérico convertido p/ string).
        top_k: Quantidade de candidatos a retornar (default: 10).

    Returns:
        DataFrame com colunas:
          - candidate_id (str)
          - nome (str)
          - score (float, 0..1)
          - skills_cv (str) — lista de skills detectadas no CV (texto livre)
        Ordenado por score desc e com no máximo `top_k` linhas.

    Raises:
        RuntimeError: Se não houver coluna de ID de vaga no schema,
                      se a vaga não tiver registros, ou se o modelo falhar.
        FileNotFoundError: Se o arquivo de modelo não existir.
    """
    logger.info("Iniciando ranking para job_id=%s (top_k=%d)", job_id, top_k)

    # 1) Carrega bundle (preprocessor + model)
    bundle = _load_bundle(cfg.model_path)
    pre = bundle["preprocessor"]
    model = bundle["model"]

    # 2) Filtra a vaga
    job_col = next((c for c in cfg.job_id_cols if c in df_view.columns), None)
    if not job_col:
        msg = f"Sem coluna de id da vaga. Colunas: {df_view.columns.tolist()}"
        logger.error(msg)
        raise RuntimeError(msg)

    df = df_view.copy()
    df[job_col] = df[job_col].astype(str)
    df = df[df[job_col] == str(job_id)].copy()
    if df.empty:
        msg = f"Vaga {job_id} sem registros na VIEW."
        logger.warning(msg)
        raise RuntimeError(msg)

    logger.debug("Registros da vaga após filtro: %d", len(df))

    # 3) Garante features e normaliza nulos
    df = _ensure_feature_columns(df, cfg)
    df.fillna("", inplace=True)

    # 4) Transforma e pontua
    try:
        X = pre.transform(df)
        proba = model.predict_proba(X)[:, 1]
    except Exception as e:
        logger.exception("Falha ao transformar/predizer: %s", e)
        raise

    # 5) Colunas de ID e nome do candidato (fallbacks seguros)
    app_col = next((c for c in cfg.applicant_id_cols if c in df.columns), None) or "_row_id"
    if app_col == "_row_id":
        logger.debug("Coluna de ID de candidato ausente; criando '_row_id'.")
        df["_row_id"] = np.arange(len(df))

    name_col = next((c for c in (cfg.col_nome_1, cfg.col_nome_2, cfg.col_nome_3) if c in df.columns), None)
    if not name_col:
        logger.debug("Coluna de nome do candidato ausente; criando '_nome_tmp'.")
        df["_nome_tmp"] = "Candidato(a)"
        name_col = "_nome_tmp"

    # 6) Monta ranking (agrega por candidato; usa média para duplicatas)
    out = (
        pd.DataFrame({
            "candidate_id": df[app_col].astype(str).values,
            "nome": df[name_col].astype(str).values,
            "score": proba,
            "skills_cv": [
                ", ".join(_skills_from_text(t))
                for t in df[cfg.col_cv_pt].astype(str).values
            ],
        })
        .groupby(["candidate_id", "nome"], as_index=False)
        .agg({"score": "mean", "skills_cv": "first"})
        .sort_values("score", ascending=False)
        .head(int(top_k))
        .reset_index(drop=True)
    )

    logger.info("Ranking concluído: %d candidatos retornados.", len(out))
    logger.debug("Top-1: %s", out.iloc[0].to_dict() if not out.empty else "sem candidatos")
    return out