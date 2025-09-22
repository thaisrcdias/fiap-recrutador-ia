from fastapi import APIRouter, HTTPException, Query
from src.llm.config import Config
from src.llm.data import DataRepository
from src.llm.llm import LLMClient
from src.llm.matcher import Matcher
from src.model_ranker import rank_top_k_for_job
from app.schemas import PredictRequest, PredictResponse, TopNResponse, TopNResponseItem

router = APIRouter()

CFG = Config()
REPO = DataRepository(CFG)
LLM = LLMClient(CFG)
MATCHER = Matcher(CFG, LLM)

@router.get("/health")
def health():
    return {"status": "ok", "project_id": CFG.project_id, "view": CFG.view_full_path}

@router.get("/top10", response_model=TopNResponse)
def top10(job_id: str = Query(...), num_candidatos: int = Query(10, ge=1, le=50)):
    try:
        topk = rank_top_k_for_job(REPO.df, CFG, job_id=job_id, top_k=int(num_candidatos))
        items = [
            TopNResponseItem(
                candidate_id=row["candidate_id"],
                nome=row["nome"],
                score=float(row["score"]),
                skills_cv=row.get("skills_cv", "")
            ) for _, row in topk.iterrows()
        ]
        return TopNResponse(job_id=str(job_id), top_n=int(num_candidatos), items=items)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    try:
        jrow = REPO.job_row(body.job_id)
        arow = REPO.applicant_row(body.applicant_id)
        vaga_text = MATCHER.job_text(jrow)
        cv_text  = MATCHER.applicant_text(arow)
        nome     = MATCHER.applicant_name(arow)
        res = MATCHER.next_action(vaga_text, cv_text, nome, k=5)
        return PredictResponse(
            job_id=str(body.job_id),
            applicant_id=str(body.applicant_id),
            nome=nome,
            score=int(res["score"]),
            justificativa=res["justificativa"],
            acao=res["acao"],
            email_text=res.get("email_text"),
            perguntas=res.get("perguntas"),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
