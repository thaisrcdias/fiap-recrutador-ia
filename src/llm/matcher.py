from __future__ import annotations

import logging
import re
from typing import Dict, Any, Iterable

from .config import Config
from .llm import LLMClient, PromptBuilder, TextUtils

logger = logging.getLogger(__name__)


class Matcher:
    """Orquestra RAG + LLM e a política de decisão para um par vaga/candidato.

    Responsabilidades:
      - Montar o texto da vaga a partir de uma linha (row) do seu dataframe/view.
      - Montar o texto do candidato (CV + campos extras) a partir da linha do dataframe.
      - Pedir ao LLM a pontuação (0–100) e as justificativas.
      - Aplicar a política:
          * score < cfg.reprova_threshold -> e-mail de agradecimento (reprova)
          * score >= cfg.reprova_threshold -> perguntas de triagem (1–2)
    """

    def __init__(self, cfg: Config, llm: LLMClient) -> None:
        """Inicializa o Matcher.

        Args:
            cfg: Configurações (nomes de colunas, thresholds, etc.).
            llm: Cliente LLM (abstrai o backend — SDK/REST).
        """
        self.cfg = cfg
        self.llm = llm
        logger.debug("Matcher inicializado (threshold=%s, triage_q=%s)",
                     self.cfg.reprova_threshold, self.cfg.triage_questions)

    # --------------------------
    # Builders de texto
    # --------------------------
    def job_text(self, job_row: Dict[str, Any]) -> str:
        """Gera o texto de contexto da vaga a partir de uma linha (row).

        Args:
            job_row: Dicionário/Series com as colunas da vaga (ex.: título, nível, requisitos).

        Returns:
            Texto normalizado para prompt do LLM.
        """
        parts: Iterable[tuple[str, str]] = [
            ("titulo_vaga", TextUtils.clean(job_row.get(self.cfg.col_titulo_vaga, ""))),
            ("nivel", TextUtils.clean(job_row.get(self.cfg.col_nivel_vaga, ""))),
            ("requisitos", TextUtils.clean(job_row.get(self.cfg.col_conh_vaga, ""))),
        ]
        text = "\n".join([f"{k}: {v}" for k, v in parts if v])
        logger.debug("job_text montado (len=%d)", len(text))
        return text

    def applicant_text(self, app_row: Dict[str, Any]) -> str:
        """Gera o texto do candidato (CV + extras) a partir de uma linha (row).

        Args:
            app_row: Dicionário/Series com colunas do candidato (CV e campos auxiliares).

        Returns:
            Texto normalizado para prompt do LLM.
        """
        cv = TextUtils.clean(app_row.get(self.cfg.col_cv_pt, ""))
        extra = " ".join([
            TextUtils.clean(app_row.get(self.cfg.col_area_atuacao, "")),
            TextUtils.clean(app_row.get(self.cfg.col_conh_cand, "")),
        ])
        text = TextUtils.clean(cv + "\n" + extra)
        logger.debug("applicant_text montado (len=%d)", len(text))
        return text

    def applicant_name(self, app_row: Dict[str, Any]) -> str:
        """Obtém o nome do candidato a partir das colunas configuradas, com fallback seguro.

        Args:
            app_row: Dicionário/Series com colunas do candidato.

        Returns:
            Nome do candidato, ou 'Candidato(a)' se ausente.
        """
        for c in (self.cfg.col_nome_1, self.cfg.col_nome_2, self.cfg.col_nome_3):
            if c in app_row and str(app_row[c]).strip():
                name = str(app_row[c]).strip()
                logger.debug("Nome do candidato encontrado em '%s': %s", c, name)
                return name
        logger.debug("Nome do candidato não encontrado; usando fallback padrão.")
        return "Candidato(a)"

    # --------------------------
    # LLM Scoring & Política
    # --------------------------
    def score(self, vaga_text: str, cv_text: str, k: int = 5) -> Dict[str, Any]:
        """Calcula a pontuação do candidato para a vaga via LLM.

        Usa RAG leve (top-k por TF-IDF) para selecionar trechos relevantes do CV
        e constrói o prompt de scoring. Em seguida, chama o LLM e normaliza a saída.

        Args:
            vaga_text: Texto da vaga (já normalizado).
            cv_text: Texto do candidato (já normalizado).
            k: Quantidade de trechos do CV para compor o contexto (default=5).

        Returns:
            Dict com:
              - score (int 0–100)
              - justificativa (List[str], até 5 itens)
        """
        selected = TextUtils.topk_by_tfidf(vaga_text, cv_text, k=k, max_chars=900)
        cv_relevant = "\n---\n".join(selected) if selected else cv_text[:1200]
        prompt = PromptBuilder.score_prompt(vaga_text, cv_relevant)

        logger.info("Chamando LLM para scoring (top_k=%d, vaga_len=%d, cv_len=%d)",
                    k, len(vaga_text), len(cv_text))
        result = self.llm.generate(prompt)
        logger.debug("Resposta LLM (score): %s", result)

        try:
            score = int(result.get("score", 0))
        except Exception:
            logger.warning("Score inválido na resposta LLM; usando 0. Resp=%s", result)
            score = 0

        just = result.get("justificativa", [])
        if isinstance(just, str):
            just = [just]
        just = [j for j in just if isinstance(j, str) and j.strip()][:5]

        score = max(0, min(100, score))
        logger.info("Score final normalizado: %d", score)
        return {"score": score, "justificativa": just}

    def next_action(self, vaga_text: str, cv_text: str, nome_cand: str, k: int = 5) -> Dict[str, Any]:
        """Aplica a política de decisão com base no score: reprova/triagem.

        Política:
          - score < cfg.reprova_threshold  -> retorna e-mail cordial de agradecimento
          - score >= cfg.reprova_threshold -> retorna 1–2 perguntas de triagem

        Args:
            vaga_text: Texto da vaga (normalizado).
            cv_text: Texto do candidato (normalizado).
            nome_cand: Nome para personalizar e-mail (fallback já tratado).
            k: Top-k chunks do CV para compor contexto de perguntas (default=5).

        Returns:
            Dict com:
              - acao: "reprovar_email" | "triagem_perguntas"
              - score: int
              - justificativa: List[str]
              - email_text (quando reprovar) OU perguntas (quando triagem)
        """
        base = self.score(vaga_text, cv_text, k=k)
        logger.info("Decidindo próxima ação (score=%d, threshold=%d)",
                    base["score"], self.cfg.reprova_threshold)

        # Reprovação
        if base["score"] < self.cfg.reprova_threshold:
            email = self._email(nome_cand, vaga_text)
            logger.info("Ação: reprovar_email (candidato=%s)", nome_cand)
            return {"acao": "reprovar_email", "email_text": email, **base}

        # Triagem (1–2 perguntas)
        selected = TextUtils.topk_by_tfidf(vaga_text, cv_text, k=k, max_chars=900)
        cv_relevant = "\n---\n".join(selected) if selected else cv_text[:900]
        q_prompt = PromptBuilder.triage_prompt(vaga_text, cv_relevant, n=self.cfg.triage_questions)

        logger.info("Chamando LLM para perguntas de triagem (n=%d)", self.cfg.triage_questions)
        q_resp = self.llm.generate(q_prompt)
        logger.debug("Resposta LLM (triagem): %s", q_resp)

        perguntas = q_resp.get("perguntas", [])
        if isinstance(perguntas, str):
            perguntas = [perguntas]
        perguntas = [p for p in perguntas if isinstance(p, str) and p.strip()][:2] or [
            "Você tem experiência prática recente com os principais requisitos da vaga? Cite exemplos.",
            "Seu nível de inglês atende ao solicitado?"
        ]

        logger.info("Ação: triagem_perguntas (qtd=%d)", len(perguntas))
        return {"acao": "triagem_perguntas", "perguntas": perguntas, **base}

    # --------------------------
    # Templates auxiliares
    # --------------------------
    def _email(self, nome: str, vaga_text: str) -> str:
        """Template do e-mail cordial de agradecimento (reprovação).

        Extrai o título da vaga do texto (se existir) para personalizar o assunto.

        Args:
            nome: Nome do candidato(a).
            vaga_text: Texto da vaga (para tentar extrair 'titulo_vaga').

        Returns:
            Corpo do e-mail em texto simples.
        """
        m = re.search(r"titulo_vaga:\s*([^\n]+)", vaga_text, re.IGNORECASE)
        titulo = m.group(1).strip() if m else "a oportunidade"
        email = (
            f"Assunto: Agradecimento pelo seu interesse\n\n"
            f"Olá, {nome},\n\n"
            f"Muito obrigado por se candidatar a {titulo}. Analisamos seu perfil e, neste momento, "
            f"seguiremos com candidaturas mais aderentes aos requisitos. "
            f"Manteremos seu contato para futuras oportunidades.\n\n"
            f"Abraços,\nEquipe de Recrutamento"
        )
        logger.debug("E-mail de reprovação gerado (len=%d)", len(email))
        return email