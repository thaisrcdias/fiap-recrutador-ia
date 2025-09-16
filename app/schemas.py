from pydantic import BaseModel

class CandidateInput(BaseModel):
    titulo_vaga: str
    nivel_vaga: str
    conhecimentos_tecnicos_vaga: str
    cv_pt: str
    area_atuacao: str
    conhecimentos_tecnicos_candidato: str
    nivel_academico_candidato: str
    nivel_ingles_candidato: str

class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    message: str
