import pandas as pd
import pytest

from src.llm.config import Config
from src.llm.data import DataRepository


class FakeLoader:
    """Falso DataLoader para simular a view do BigQuery."""
    def __init__(self, project_id, view_full_path, df=None):
        self.project_id = project_id
        self.view_full_path = view_full_path
        self._df = df

    def load_data(self):
        return self._df


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def df_view() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "id_vaga": "5180",
            "titulo_vaga": "Engenheiro de Dados",
            "nivel_vaga": "Sênior",
            "conhecimentos_tecnicos_vaga": "Python, SQL, BigQuery",
            "id_applicant": "A1",
            "nome_candidato": "Ana Silva",
            "email": "ana@example.com",
            "cv_pt": "Experiência com Python e SQL.",
            "area_atuacao": "Dados",
            "conhecimentos_tecnicos_candidato": "Python, SQL"
        },
        {
            "id_vaga": "5180",
            "titulo_vaga": "Engenheiro de Dados",
            "nivel_vaga": "Sênior",
            "conhecimentos_tecnicos_vaga": "Python, SQL, BigQuery",
            "id_applicant": "A2",
            "nome_candidato": "Bruno Lima",
            "email": "bruno@example.com",
            "cv_pt": "Trabalhou com BigQuery e Spark.",
            "area_atuacao": "Dados",
            "conhecimentos_tecnicos_candidato": "BigQuery, Spark"
        },
        {
            "id_vaga": "777",
            "titulo_vaga": "Analista",
            "nivel_vaga": "Pleno",
            "conhecimentos_tecnicos_vaga": "Excel",
            "id_applicant": "X1",
            "nome_candidato": "Fulano",
            "email": "fulano@example.com",
            "cv_pt": "Planilhas",
            "area_atuacao": "Backoffice",
            "conhecimentos_tecnicos_candidato": "Excel"
        },
    ])


def test_data_repository_loads_with_dataloader(monkeypatch, cfg, df_view):
    """Garante que _load_view_df usa o DataLoader quando disponível."""
    # Monkeypatch do DataLoader usado dentro do módulo
    from src.llm import data as data_module

    def fake_loader_ctor(project_id, view_full_path):
        return FakeLoader(project_id, view_full_path, df=df_view)

    monkeypatch.setattr(data_module, "DataLoader", fake_loader_ctor)

    repo = DataRepository(cfg)
    assert not repo.df.empty
    assert len(repo.df) == 3


def test_data_repository_fails_on_empty_view(monkeypatch, cfg):
    """Quando a view está vazia, dispara RuntimeError."""
    from src.llm import data as data_module

    def fake_loader_ctor(project_id, view_full_path):
        return FakeLoader(project_id, view_full_path, df=pd.DataFrame())

    monkeypatch.setattr(data_module, "DataLoader", fake_loader_ctor)

    with pytest.raises(RuntimeError, match="View vazia ou inacessível"):
        DataRepository(cfg)


def test_job_row_ok(monkeypatch, cfg, df_view):
    """job_row retorna a primeira linha da vaga."""
    from src.llm import data as data_module
    monkeypatch.setattr(data_module, "DataLoader", lambda *a, **k: FakeLoader(None, None, df=df_view))
    repo = DataRepository(cfg)

    row = repo.job_row("5180")
    assert row["titulo_vaga"] == "Engenheiro de Dados"
    assert row["id_vaga"] == "5180"


def test_job_row_not_found(monkeypatch, cfg, df_view):
    from src.llm import data as data_module
    monkeypatch.setattr(data_module, "DataLoader", lambda *a, **k: FakeLoader(None, None, df=df_view))
    repo = DataRepository(cfg)

    with pytest.raises(RuntimeError, match="não encontrada"):
        repo.job_row("NAO-EXISTE")


def test_applicants_for_job(monkeypatch, cfg, df_view):
    from src.llm import data as data_module
    monkeypatch.setattr(data_module, "DataLoader", lambda *a, **k: FakeLoader(None, None, df=df_view))
    repo = DataRepository(cfg)

    sub = repo.applicants_for_job("5180")
    assert len(sub) == 2
    assert set(sub["id_applicant"]) == {"A1", "A2"}


def test_applicant_row_by_id(monkeypatch, cfg, df_view):
    from src.llm import data as data_module
    monkeypatch.setattr(data_module, "DataLoader", lambda *a, **k: FakeLoader(None, None, df=df_view))
    repo = DataRepository(cfg)

    row = repo.applicant_row("A2")
    assert row["nome_candidato"] == "Bruno Lima"
    assert row["email"] == "bruno@example.com"


def test_applicant_row_fallback_email(monkeypatch, cfg, df_view):
    """Se ID não bater, tenta por e-mail."""
    df = df_view.copy()
    df["id_applicant"] = ["", "", "X1"]
    from src.llm import data as data_module
    monkeypatch.setattr(data_module, "DataLoader", lambda *a, **k: FakeLoader(None, None, df=df))
    repo = DataRepository(cfg)

    row = repo.applicant_row("bruno@example.com")
    assert row["nome_candidato"] == "Bruno Lima"


def test_applicant_row_fallback_name(monkeypatch, cfg, df_view):
    """Se ID/e-mail não bater, tenta por nome."""
    df = df_view.copy()
    df["id_applicant"] = ["", "", ""]
    df["email"] = ["", "", ""]
    from src.llm import data as data_module
    monkeypatch.setattr(data_module, "DataLoader", lambda *a, **k: FakeLoader(None, None, df=df))
    repo = DataRepository(cfg)

    row = repo.applicant_row("ana silva")
    assert row["nome_candidato"] == "Ana Silva"


def test_applicant_row_not_found(monkeypatch, cfg, df_view):
    from src.llm import data as data_module
    monkeypatch.setattr(data_module, "DataLoader", lambda *a, **k: FakeLoader(None, None, df=df_view))
    repo = DataRepository(cfg)

    with pytest.raises(RuntimeError, match="não encontrado"):
        repo.applicant_row("inexistente")
