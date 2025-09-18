
import os
import requests

API_KEY = ""  # exportada no Cloud Shell
MODEL = "gemini-2.5-flash-lite"
URL = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{MODEL}:streamGenerateContent?key={API_KEY}"


vaga = "Engenheiro de Dados"
nivel = "Sênior"
conhecimentos = "Python, SQL, BigQuery, Spark"

prompt = f"""
Você é um recrutador de tecnologia.
Gere 3 perguntas de entrevista para um candidato a {vaga}, nível {nivel},
considerando que precisa ter experiência em {conhecimentos}.
"""

payload = {
    "contents": [
        {
            "role": "user",
            "parts": [
                {"text": prompt}
            ]
        }
    ]
}

headers = {"Content-Type": "application/json"}

response = requests.post(URL, headers=headers, json=payload)

print("Status:", response.status_code)
print("Response:", response.text)