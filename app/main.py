from fastapi import FastAPI
from app.routes import router

app = FastAPI(
    title="Recruitment Prediction API",
    description="API para predição de encaminhamento de candidatos",
    version="1.0.0",
)

app.include_router(router)

@app.get("/")
def root():
    return {"message": "Recruitment API rodando."}