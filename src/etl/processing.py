import json
import pandas as pd
from google.cloud import storage
from google.cloud import bigquery

# Executar no VScode
# $env:GOOGLE_APPLICATION_CREDENTIALS="C:\Users\thais.r.carvalho\Documents\Pos Machine Learning\Módulo 5\fiap-recrutador-ia\resolute-spirit-472116-f2-a11e1f97fbf0.json"
# $env:PROJECT_ID="resolute-spirit-472116-f2" 


# --- CONFIGURAÇÕES ---
GCS_BUCKET_NAME = 'recrutamento-decision'
VAGAS_FILE_PATH = 'vagas/vagas.json'
PROSPECTS_FILE_PATH = 'prospects/prospects.json'
APPLICANTS_FILE_PATH = 'applicants/applicants.json'
BIGQUERY_PROJECT_ID = 'resolute-spirit-472116-f2'
BIGQUERY_DATASET_ID = 'recrutamento'
BIGQUERY_VAGAS_TABLE_ID = 'vagas'
BIGQUERY_PROSPECTS_TABLE_ID = 'prospects'
BIGQUERY_APPLICANTS_TABLE_ID = 'applicants'
BIGQUERY_VIEW_ID = 'vw_vagas_prospects_applicants_completo'
# Configurações do BigQuery
WRITE_DISPOSITION = "WRITE_TRUNCATE"

def process_and_load_data():
    """
    Função principal que orquestra a leitura, processamento e carga de
    todos os arquivos JSON do GCS para o BigQuery.
    """
    storage_client = storage.Client()
    bq_client = bigquery.Client(project=BIGQUERY_PROJECT_ID)
    bucket = storage_client.bucket(GCS_BUCKET_NAME)

    # --- 1. Processar e Carregar VAGAS ---
    print("Iniciando processamento de vagas.json...")
    vagas_blob = bucket.blob(VAGAS_FILE_PATH)
    vagas_data = json.loads(vagas_blob.download_as_string())
    
    vagas_list = []
    for vaga_id, data in vagas_data.items():
        flat_dict = {
            'id_vaga': vaga_id,
            **data.get('informacoes_basicas', {}),
            **data.get('perfil_vaga', {}),
            **data.get('beneficios', {})
        }

        if 'nivel profissional' in flat_dict:
            flat_dict['nivel_profissional'] = flat_dict.pop('nivel profissional')
        vagas_list.append(flat_dict)
    
    vagas_df = pd.DataFrame(vagas_list)
    load_table_from_dataframe(bq_client, vagas_df, 'vagas')

    # --- 2. Processar e Carregar PROSPECTS ---
    print("\nIniciando processamento de prospects.json...")
    prospects_blob = bucket.blob(PROSPECTS_FILE_PATH)
    prospects_data = json.loads(prospects_blob.download_as_string())

    prospects_list = []
    for vaga_id, data in prospects_data.items():
        for prospect in data.get('prospects', []):
            prospect['id_vaga'] = vaga_id
            prospects_list.append(prospect)
            
    prospects_df = pd.DataFrame(prospects_list)
    load_table_from_dataframe(bq_client, prospects_df, 'prospects')

    # --- 3. Processar e Carregar APPLICANTS ---
    print("\nIniciando processamento de applicants.json...")
    applicants_blob = bucket.blob(APPLICANTS_FILE_PATH)
    applicants_data = json.loads(applicants_blob.download_as_string())

    applicants_list = []
    for applicant_id, data in applicants_data.items():
        flat_dict = {
            'id_applicant': applicant_id,
            **data.get('infos_basicas', {}),
            **data.get('informacoes_pessoais', {}),
            **data.get('informacoes_profissionais', {}),
            **data.get('formacao_e_idiomas', {}),
            **data.get('cargo_atual', {}),
            'cv_pt': data.get('cv_pt'),
            'cv_en': data.get('cv_en')
        }
        applicants_list.append(flat_dict)

    applicants_df = pd.DataFrame(applicants_list)
    load_table_from_dataframe(bq_client, applicants_df, 'applicants')

    print("\nRecriando a VIEW consolidada no BigQuery...")
    create_consolidated_view(bq_client)
    
    print("\nProcesso de ETL concluído com sucesso!")


def load_table_from_dataframe(bq_client, df: pd.DataFrame, table_name: str):
    """
    Carrega um DataFrame para uma tabela no BigQuery, usando um schema auto-gerado.
    """
    table_full_path = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.{table_name}"
    
    # Limpeza de caracteres inválidos nos nomes das colunas
    df.columns = df.columns.str.replace('[^0-9a-zA-Z_]', '', regex=True)

    job_config = bigquery.LoadJobConfig(
        write_disposition=WRITE_DISPOSITION,
        autodetect=True
    )

    try:
        job = bq_client.load_table_from_dataframe(df, table_full_path, job_config=job_config)
        job.result()
        print(f"Tabela '{table_full_path}' carregada com {len(df)} registros.")
    except Exception as e:
        print(f"Erro ao carregar a tabela '{table_full_path}': {e}")


def create_consolidated_view(bq_client):
    """
    Cria ou substitui a VIEW que une as três tabelas principais.
    """
    dataset_ref = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}"
    view_full_path = f"{dataset_ref}.vagas_prospects_applicants_completo"

    query = f"""
    CREATE OR REPLACE VIEW `{view_full_path}` AS
    SELECT
      -- VAGAS
      v.id_vaga,
      v.titulo_vaga,
      v.cliente,
      v.tipo_contratacao,
      v.nivel_profissional AS nivel_vaga,
      v.nivel_academico AS nivel_academico_vaga,
      v.nivel_ingles AS nivel_ingles_vaga,
      v.principais_atividades,
      v.competencia_tecnicas_e_comportamentais AS conhecimentos_tecnicos_vaga,

      -- PROSPECTS
      p.codigo AS codigo_prospecto,
      p.situacao_candidado AS situacao_candidato,
      p.data_candidatura,
      p.recrutador,

      -- APPLICANTS
      a.id_applicant,
      a.nome AS nome_candidato,
      a.email,
      a.telefone,
      a.url_linkedin,
      a.area_atuacao,
      a.nivel_academico AS nivel_academico_candidato,
      a.nivel_ingles AS nivel_ingles_candidato,
      a.conhecimentos_tecnicos AS conhecimentos_tecnicos_candidato,
      a.cv_pt
    FROM
      `{dataset_ref}.vagas` AS v
    JOIN
      `{dataset_ref}.prospects` AS p ON v.id_vaga = p.id_vaga
    JOIN
      `{dataset_ref}.applicants` AS a ON p.codigo = a.id_applicant
    """

    try:
        query_job = bq_client.query(query)
        query_job.result()
        print(f"VIEW '{view_full_path}' criada/atualizada com sucesso.")
    except Exception as e:
        print(f"Erro ao criar a VIEW: {e}")


if __name__ == "__main__":
    process_and_load_data()