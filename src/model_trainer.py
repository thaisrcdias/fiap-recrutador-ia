import xgboost as xgb
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

class ModelTrainer:
    """Treina, avalia e salva o modelo de Machine Learning."""

    def __init__(self, model_path: str = 'saved_models/screening_model.joblib'):
        self.model_path = model_path
        self.model = xgb.XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss', 
            random_state=42
        )
        self.preprocessor = None

    def train(self, X, y):
        """Treina o modelo com os dados pré-processados."""
        if y.nunique() < 2:
            print("ERRO: A variável alvo possui apenas uma classe. O treinamento não pode continuar.")
            return

        num_negativos = (y == 0).sum()
        num_positivos = (y == 1).sum()
        
        if num_positivos > 0:
            scale_pos_weight = num_negativos / num_positivos
            self.model.set_params(scale_pos_weight=scale_pos_weight)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("\nIniciando o treinamento do modelo XGBoost...")
        self.model.fit(X_train, y_train)
        
        self.evaluate(X_test, y_test)

    def evaluate(self, X_test, y_test):
        """Avalia o modelo e imprime as métricas."""
        y_pred = self.model.predict(X_test)
        
        print("\n--- Avaliação do Modelo (Prevendo 'Encaminhado ao Requisitante') ---")
        
        if len(np.unique(y_test)) > 1:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            print(f"AUC-ROC Score: {auc_score:.4f}")
            if auc_score < 0.7:
                print("AVISO: Performance do modelo abaixo do limiar aceitável.")
            else:
                print("Modelo com performance aceitável para produção.")
        else:
            print("AUC-ROC Score: Não pôde ser calculado (apenas uma classe no conjunto de teste).")
        
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=['Não Encaminhado', 'Encaminhado']))
        print("--------------------------\n")

    # def save_pipeline(self, preprocessor_pipeline):
    #     """Salva o pipeline de pré-processamento e o modelo juntos."""
    #     full_pipeline = {
    #         'preprocessor': preprocessor_pipeline,
    #         'model': self.model
    #     }
    #     joblib.dump(full_pipeline, "saved_models/match_model.joblib")
    #     print("Pipeline completo salvo em: saved_models/match_model.joblib")


    def save_pipeline(self, preprocessor_pipeline):
        import os
        import tempfile
        """
        Salva o pipeline completo (preprocessador + modelo).
        - Se model_path começar com 'gs://', salva localmente em arquivo temporário e faz upload ao GCS.
        - Caso contrário, salva em disco local criando a pasta se necessário.
        """
        if self.model is None:
            raise RuntimeError("Modelo ainda não treinado. Chame .train() antes de salvar.")

        full_pipeline = {
            "preprocessor": preprocessor_pipeline,
            "model": self.model,
        }

        dest = self.model_path
        if dest.startswith("gs://"):
            # 1) salva em temp local
            suffix = os.path.splitext(dest)[-1] or ".joblib"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)

            print("Salvando bundle temporário em: %s", tmp_path)
            joblib.dump(full_pipeline, tmp_path)

            # 2) faz upload para GCS
            from google.cloud import storage
            _, _, bucket_name, *blob_parts = dest.split("/")
            blob_path = "/".join(blob_parts)
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            print("Enviando artefato para GCS: gs://%s/%s", bucket_name, blob_path)
            blob.upload_from_filename(tmp_path)
            os.remove(tmp_path)
            print("Bundle salvo no GCS em: %s", dest)
            return dest
        else:
            # salvar localmente
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            print("Salvando bundle local em: %s", dest)
            joblib.dump(full_pipeline, dest)
            print("Bundle salvo em: %s", dest)
            return dest