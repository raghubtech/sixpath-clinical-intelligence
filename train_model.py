from pathlib import Path
import json,hashlib,pandas as pd
from src.pipeline import read_dataset,train_and_evaluate,save_bundle
ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"realistic_multi_organ_diabetes_dataset_4000.csv"
MODEL=ROOT/"artifacts"/"risk_model.joblib"
METRICS=ROOT/"artifacts"/"evaluation_metrics.json"
if __name__=="__main__":
 df=read_dataset(DATA); bundle=train_and_evaluate(df); bundle["data_signature"]=hashlib.sha256(DATA.read_bytes()).hexdigest()
 save_bundle(bundle,MODEL); METRICS.write_text(json.dumps(bundle["metrics"],indent=2),encoding="utf-8")
 print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
 print(f"Saved model: {MODEL}"); print(pd.DataFrame(bundle["metrics"]).to_string(index=False))
