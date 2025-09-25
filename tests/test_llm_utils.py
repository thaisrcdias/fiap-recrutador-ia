import json
import re
import pytest

from src.llm.llm import TextUtils, PromptBuilder, LLMClient
from src.llm.config import Config


def test_textutils_clean_and_split():
    txt = "  Olá   mundo \n\n com  espaços \nquebrados.  "
    cleaned = TextUtils.clean(txt)
    assert cleaned == "Olá mundo com espaços quebrados."

    chunks = TextUtils.split_chunks("a b. c d\ne f", max_chars=4)

    assert all(len(c) <= 6 for c in chunks)
    assert isinstance(chunks, list)


def test_textutils_topk_by_tfidf_basic():
    q = "python sql bigquery"
    doc = "Tenho experiência com BigQuery e Python. Também já usei Spark e SQL."
    top = TextUtils.topk_by_tfidf(q, doc, k=2, max_chars=50)
    assert 1 <= len(top) <= 2
    assert any("Python" in c or "BigQuery" in c or "SQL" in c for c in top)


def test_promptbuilder_contains_schema():
    p = PromptBuilder.score_prompt("vaga x", "cv y")
    assert "Responda APENAS em JSON válido" in p
    assert "score" in p and "justificativa" in p

    pt = PromptBuilder.triage_prompt("vaga", "cv", n=2)
    assert '{"perguntas":' in pt


def test_llmclient_force_json_ok():
    raw = json.dumps({"score": 77, "justificativa": ["ok"]})
    out = LLMClient._force_json(raw)
    assert out["score"] == 77


def test_llmclient_force_json_extracts_from_noise():
    raw = "bla bla {\"score\": 45, \"justificativa\": [\"x\"]} fim"
    out = LLMClient._force_json(raw)
    assert out["score"] == 45


def test_llmclient_generate_uses_backend(monkeypatch):
    cfg = Config()
    client = LLMClient(cfg)

    called = {"sdk": 0, "rest": 0}

    def fake_sdk(prompt: str):
        called["sdk"] += 1
        return {"score": 80, "justificativa": ["ok"]}

    def fake_rest(prompt: str):
        called["rest"] += 1
        return {"score": 70, "justificativa": ["ok"]}

    monkeypatch.setattr(LLMClient, "_call_genai_sdk", staticmethod(fake_sdk))
    monkeypatch.setattr(LLMClient, "_call_rest", staticmethod(fake_rest))

    # Caso 1: backend = genai_sdk, sem exceção
    cfg1 = Config()
    object.__setattr__(cfg1, "llm_backend", "genai_sdk")
    out1 = LLMClient(cfg1).generate("prompt")
    assert out1["score"] == 80
    assert called["sdk"] == 1

    # Caso 2: backend = genai_sdk, sdk falha, cai no REST
    called["sdk"] = 0
    called["rest"] = 0

    def raise_err(prompt: str):
        called["sdk"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(LLMClient, "_call_genai_sdk", staticmethod(raise_err))
    out2 = LLMClient(cfg1).generate("prompt")
    assert out2["score"] == 70
    assert called["sdk"] == 1 and called["rest"] == 1

    # Caso 3: backend = rest_publisher
    cfg2 = Config()
    object.__setattr__(cfg2, "llm_backend", "rest_publisher")
    out3 = LLMClient(cfg2).generate("prompt")
    assert out3["score"] == 70
