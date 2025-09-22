from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass(frozen=True)
class Config:
    """Config central da aplicação (env-first)."""
    project_id: str = os.getenv("PROJECT_ID", "resolute-spirit-472116-f2")
    view_full_path: str = os.getenv(
        "VIEW_FULL_PATH",
        "resolute-spirit-472116-f2.recrutamento.vagas_prospects_applicants_completo",
    )

    # Caminho do bundle do modelo clássico (preprocessor + modelo) salvo no treino
    model_path: str = os.getenv("MODEL_PATH", "app/saved_models/match_model.joblib")

    # LLM
    llm_backend: str = os.getenv("LLM_BACKEND", "genai_sdk")  # "genai_sdk" | "rest_publisher"
    model_sdk: str = os.getenv("GENAI_MODEL", "gemini-2.0-pro-exp-02-05")
    model_rest: str = os.getenv("GEMINI_MODEL_REST", "gemini-2.5-flash-lite")
    api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")

    # Colunas tolerantes ao schema da VIEW
    job_id_cols: Tuple[str, ...] = ("job_id", "id_vaga", "vaga_id", "codigo_vaga", "id")
    applicant_id_cols: Tuple[str, ...] = ("id_applicant", "applicant_id", "codigo_profissional", "codigo", "id_candidato", "id")

    # Conteúdo
    col_titulo_vaga: str = "titulo_vaga"
    col_nivel_vaga: str = "nivel_vaga"
    col_conh_vaga: str = "conhecimentos_tecnicos_vaga"

    col_cv_pt: str = "cv_pt"
    col_area_atuacao: str = "area_atuacao"
    col_conh_cand: str = "conhecimentos_tecnicos_candidato"

    col_nome_1: str = "nome_candidato"
    col_nome_2: str = "nome"
    col_nome_3: str = "nome_completo"

    # Política
    reprova_threshold: int = int(os.getenv("REPROVA_THRESHOLD", "45"))
    triage_questions: int = int(os.getenv("TRIAGE_QUESTIONS", "2"))
