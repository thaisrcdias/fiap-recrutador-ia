import pandas as pd
from google.cloud import bigquery


class DataLoader:
    """Carrega os dados da view consolidada no BigQuery."""

    def __init__(self, project_id: str, view_full_path: str):
        self.project_id = project_id
        self.view_full_path = view_full_path
        # Usa credenciais do ambiente (GOOGLE_APPLICATION_CREDENTIALS)
        self.client = bigquery.Client(project=self.project_id)

    def load_data(self) -> pd.DataFrame:
        print(f"Carregando dados de: {self.view_full_path}...")
        query = f"SELECT * FROM `{self.view_full_path}`"
        try:
            df = self.client.query(query).to_dataframe()
            print(f"Dados carregados com sucesso: {df.shape[0]} registros, {df.shape[1]} colunas.")
            return df
        except Exception as e:
            print(f"Erro ao carregar dados do BigQuery: {e}")
            return pd.DataFrame()
