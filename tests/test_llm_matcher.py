import pytest

from src.llm.config import Config
from src.llm.matcher import Matcher
from src.llm.llm import LLMClient


class DummyLLM(LLMClient):
    """LLM determinístico: retorna respostas previsíveis para facilitar teste."""
    def __init__(self, cfg, score=50, perguntas=None):
        super().__init__(cfg)
        self._score = score
        self._perguntas = perguntas or ["P1", "P2"]

    def generate(self, prompt: str):
        # Heurística: se o prompt pedir perguntas (tem a palavra-chave do PromptBuilder)
        if '{"perguntas":' in prompt:
            return {"perguntas": self._perguntas}
        return {"score": self._score, "justificativa": ["ok", "coerente"]}


@pytest.fixture
def cfg() -> Config:
    # threshold 45 por padrão (vem do dataclass)
    return Config()


def test_matcher_job_and_applicant_text(cfg):
    llm = DummyLLM(cfg)
    m = Matcher(cfg, llm)

    job_row = {
        cfg.col_titulo_vaga: "Engenheiro de Dados",
        cfg.col_nivel_vaga: "Sênior",
        cfg.col_conh_vaga: "Python, SQL, BigQuery",
    }
    app_row = {
        cfg.col_cv_pt: "Experiência com Python e SQL. Projetos em dados.",
        cfg.col_area_atuacao: "Dados",
        cfg.col_conh_cand: "Python, SQL, BigQuery",
        cfg.col_nome_1: "Ana Silva"
    }

    jtxt = m.job_text(job_row)
    atxt = m.applicant_text(app_row)
    name = m.applicant_name(app_row)

    assert "titulo_vaga" in jtxt and "Engenheiro" in jtxt
    assert "requisitos" in jtxt and "Python" in jtxt
    assert "Dados" in atxt and "Projetos" in atxt
    assert name == "Ana Silva"


def test_matcher_score_parsing(cfg):
    llm = DummyLLM(cfg, score=88)
    m = Matcher(cfg, llm)

    out = m.score("vaga text", "cv text", k=3)
    assert 0 <= out["score"] <= 100
    assert isinstance(out["justificativa"], list)


def test_matcher_next_action_reprova(cfg):
    # Força score baixo para cair em reprovação
    llm = DummyLLM(cfg, score=20)
    m = Matcher(cfg, llm)

    res = m.next_action("titulo_vaga: Dev\nrequisitos: X", "cv", "Candidato X", k=2)
    assert res["acao"] == "reprovar_email"
    assert "Agradecimento" in res["email_text"]


def test_matcher_next_action_triage(cfg):
    # Score >= threshold -> triagem
    llm = DummyLLM(cfg, score=75, perguntas=["Pergunta A", "Pergunta B"])
    m = Matcher(cfg, llm)

    res = m.next_action("vaga", "cv", "Nome", k=2)
    assert res["acao"] == "triagem_perguntas"
    assert res["perguntas"] == ["Pergunta A", "Pergunta B"]
