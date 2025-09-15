
import os, datetime, subprocess
from google.cloud import aiplatform

PROJECT_ID = os.getenv("PROJECT_ID", "resolute-spirit-472116-f2")
REGION = os.getenv("REGION", "us-central1")
BUCKET = os.getenv("BUCKET", f"{PROJECT_ID}-mlops-artifacts")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "saved_models/match_model.joblib")
MODEL_NAME = os.getenv("MODEL_NAME", "recrutador-match")

timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
GCS_DIR = f"gs://{BUCKET}/models/{MODEL_NAME}/{timestamp}/"

# 1) upload
subprocess.run(["gsutil","cp",LOCAL_MODEL_PATH,GCS_DIR], check=True)

# 2) registry
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET}")
model = aiplatform.Model.upload(
    display_name=MODEL_NAME,
    artifact_uri=GCS_DIR,
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
)
print("Modelo registrado:", model.resource_name)
print("Artefato:", GCS_DIR)
