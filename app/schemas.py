from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional

class PredictRequest(BaseModel):
    job_id: str = Field(..., description="ID da vaga (ex.: 5180)")
    applicant_id: str = Field(..., description="ID do candidato (id_applicant) ou e-mail/nome")

class PredictResponse(BaseModel):
    job_id: str
    applicant_id: str
    nome: str
    score: int
    justificativa: List[str]
    acao: str
    email_text: Optional[str] = None
    perguntas: Optional[List[str]] = None

class TopNResponseItem(BaseModel):
    candidate_id: str
    nome: str
    score: float
    skills_cv: str

class TopNResponse(BaseModel):
    job_id: str
    top_n: int
    items: List[TopNResponseItem]
