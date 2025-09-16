from fastapi import APIRouter, HTTPException
from app.schemas import CandidateInput, PredictionOutput
from app.predictor import Predictor

router = APIRouter()
predictor = Predictor()

@router.post("/predict", response_model=PredictionOutput)
def predict(candidate: CandidateInput):
    """
    Endpoint que recebe dados de um candidato e retorna predição do modelo.
    """
    try:
        result = predictor.predict(candidate.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
