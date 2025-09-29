from __future__ import annotations
import re, json, requests
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from .config import Config

class TextUtils:
    """Limpeza e RAG por TF-IDF."""
    @staticmethod
    def clean(text: str) -> str:
        if not text: return ""
        import re
        return re.sub(r"\s+", " ", str(text).strip())

    @staticmethod
    def split_chunks(text: str, max_chars: int = 900) -> List[str]:
        if not text: return []
        import re
        parts = re.split(r"\n\n|\n|\. ", text)
        chunks, buf = [], ""
        for p in parts:
            p = p.strip()
            if not p: continue
            if len(buf) + len(p) + 1 <= max_chars:
                buf = (buf + " " + p).strip()
            else:
                if buf: chunks.append(buf)
                buf = p
        if buf: chunks.append(buf)
        return chunks

    @staticmethod
    def topk_by_tfidf(query: str, doc: str, k: int = 5, max_chars: int = 900) -> List[str]:
        chunks = TextUtils.split_chunks(doc, max_chars=max_chars)
        if not chunks: return []
        corpus = [query] + chunks
        vec = TfidfVectorizer(max_features=4000, ngram_range=(1,2))
        X = vec.fit_transform(corpus)
        q = X[0]; D = X[1:]
        sims = (D @ q.T).toarray().ravel()
        idx = np.argsort(-sims)[:k]
        return [chunks[i] for i in idx]

class PromptBuilder:
    """Prompts padronizados."""
    @staticmethod
    def score_prompt(vaga_text: str, cv_relevant_text: str) -> str:
        schema = """
                    Responda APENAS em JSON válido:
                    {
                    "score": <int 0-1000>,
                    "justificativa": ["bullet", "bullet", "bullet"]
                    }
                    """
        return f"""
                    Você é um especialista em RH Tech.  
                    
                    Analise se o candidato possui perfil técnico para vaga. 
                    As informações do currículo do candidato devem se parecidas com a descrição da vaga.
                    Dê mais peso para tempo de experiência na área da vaga.  

                    Vaga:
                    {vaga_text}

                    Trechos do currículo:
                    {cv_relevant_text}

                    Tarefa:
                    Atribua um score 0–1000, sendo 0 para não aderente a vaga e 1000 muito aderente a vaga. Avalie aderência geral do candidato à vaga com justificativas curtas.
                    {schema}
                """.strip()

    @staticmethod
    def triage_prompt(vaga_text: str, cv_relevant_text: str, n: int = 2) -> str:
        n = max(1, min(2, int(n)))
        return f"""
                Você é um recrutador técnico.
                Gere {n/2} perguntas objetivas de triagem (confirmar experiência/idioma quando aplicável).
                Gere também {n/2} perguntas sobre o perfil técnico da vaga. Monte uma sabatina de {n/2} perguntas para o candidato com exemplos de código.

                Vaga:
                {vaga_text}

                Trechos do CV:
                {cv_relevant_text}

                Responda APENAS em JSON:
                {{"perguntas": ["...", "..."]}}
                """.strip()

class LLMClient:
    """Encapsula chamada ao Gemini (SDK/REST)."""
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def generate(self, prompt: str) -> Dict[str, Any]:
        if self.cfg.llm_backend == "genai_sdk":
            try:
                return self._call_genai_sdk(prompt)
            except Exception:
                return self._call_rest(prompt)
        return self._call_rest(prompt)

    def _call_genai_sdk(self, prompt: str) -> Dict[str, Any]:
        if not self.cfg.api_key:
            raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY ausente.")
        from google import genai
        client = genai.Client(api_key=self.cfg.api_key)
        resp = client.models.generate_content(model=self.cfg.model_sdk, contents=[prompt])
        raw = getattr(resp, "text", "") or ""
        return self._force_json(raw)

    def _call_rest(self, prompt: str) -> Dict[str, Any]:
        if not self.cfg.api_key:
            raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY ausente.")
        url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{self.cfg.model_rest}:generateContent?key={self.cfg.api_key}"
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            text = json.dumps(data, ensure_ascii=False)
        return self._force_json(text)

    @staticmethod
    def _force_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            import re, json as _json
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                return {"error": "Falha ao interpretar JSON", "raw": text}
            try:
                return _json.loads(m.group(0))
            except Exception:
                return {"error": "Falha ao interpretar JSON", "raw": text}
