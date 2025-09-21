from __future__ import annotations
import os
import pandas as pd
from typing import Optional, Tuple
from .config import Config
import logging
from src.data_loader import DataLoader 

logger = logging.getLogger(__name__)

class DataRepository:
    """Acesso à VIEW do BigQuery e seleção de linhas de vaga/candidato."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.df = self._load_view_df()

    def _load_view_df(self) -> pd.DataFrame:
        """Carrega via DataLoader (se existir) ou via BigQuery client (fallback)."""
        try:
            logger.info("Carregando via src.data_loader.DataLoader")
            loader = DataLoader(self.cfg.project_id, self.cfg.view_full_path)
            df = loader.load_data()
        except Exception:
            logger.info("Fallback: carregando via google-cloud-bigquery")
            from google.cloud import bigquery
            client = bigquery.Client(project=self.cfg.project_id)
            df = client.query(f"SELECT * FROM `{self.cfg.view_full_path}`").to_dataframe()

        if df is None or df.empty:
            raise RuntimeError("View vazia ou inacessível.")
        logger.info("VIEW carregada: %d linhas / %d colunas", len(df), len(df.columns))
        return df

    def _pick_col(self, candidates: Tuple[str, ...]) -> Optional[str]:
        for c in candidates:
            if c in self.df.columns:
                return c
        return None

    def job_row(self, job_id: str) -> pd.Series:
        col = self._pick_col(self.cfg.job_id_cols)
        if not col:
            raise RuntimeError(f"Sem coluna de id da vaga. Colunas: {self.df.columns.tolist()}")
        dfx = self.df.copy()
        dfx[col] = dfx[col].astype(str)
        sub = dfx[dfx[col] == str(job_id)]
        if sub.empty:
            raise RuntimeError(f"Vaga {job_id} não encontrada na coluna {col}.")
        return sub.iloc[0]

    def applicants_for_job(self, job_id: str) -> pd.DataFrame:
        col = self._pick_col(self.cfg.job_id_cols)
        if not col:
            raise RuntimeError(f"Sem coluna de id da vaga. Colunas: {self.df.columns.tolist()}")
        dfx = self.df.copy()
        dfx[col] = dfx[col].astype(str)
        return dfx[dfx[col] == str(job_id)].copy()

    def applicant_row(self, applicant_id: str) -> pd.Series:
        def _norm(x): return str(x).strip().lower()
        col = self._pick_col(self.cfg.applicant_id_cols)
        if col:
            dfx = self.df.copy()
            dfx[col] = dfx[col].astype(str)
            sub = dfx[dfx[col].apply(_norm) == _norm(applicant_id)]
            if not sub.empty:
                return sub.iloc[0]

        if "email" in self.df.columns and "@" in str(applicant_id):
            sub = self.df[self.df["email"].astype(str).str.strip().str.lower() == _norm(applicant_id)]
            if not sub.empty:
                return sub.iloc[0]

        for name_col in (self.cfg.col_nome_1, self.cfg.col_nome_2, self.cfg.col_nome_3):
            if name_col in self.df.columns:
                sub = self.df[self.df[name_col].astype(str).str.strip().str.lower() == _norm(applicant_id)]
                if not sub.empty:
                    return sub.iloc[0]

        raise RuntimeError(
            f"Candidato '{applicant_id}' não encontrado por ID/e-mail/nome. "
            f"Colunas: {self.df.columns.tolist()} | IDs: {self.cfg.applicant_id_cols}"
        )
