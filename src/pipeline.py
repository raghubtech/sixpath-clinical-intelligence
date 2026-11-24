"""Training, evaluation and inference for six binary diabetic-complication outcomes."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

TARGETS = ["Nephropathy_Risk", "Cardiovascular_Risk", "Hepatic_Risk", "Neuropathy_Risk", "Retinopathy_Risk", "Cerebrovascular_Risk"]
DISPLAY_NAMES = {
 "Nephropathy_Risk":"Diabetic nephropathy", "Cardiovascular_Risk":"Cardiovascular disease",
 "Hepatic_Risk":"Hepatic complications", "Neuropathy_Risk":"Diabetic neuropathy",
 "Retinopathy_Risk":"Diabetic retinopathy", "Cerebrovascular_Risk":"Cerebrovascular disease"
}

def read_dataset(path):
    df = pd.read_csv(path)
    if "Cardiovascular_Risk" not in df and "Cardio_Risk" in df:
        df = df.rename(columns={"Cardio_Risk":"Cardiovascular_Risk"})
    missing = [c for c in TARGETS if c not in df]
    if missing: raise ValueError("Missing required target columns: " + ", ".join(missing))
    return df

def split_features_targets(df):
    X = df.drop(columns=TARGETS).copy()
    y = df[TARGETS].copy()
    id_cols = [c for c in X if c.strip().lower() in {"patient_id","record_id","id","index"}]
    if id_cols: X = X.drop(columns=id_cols)
    for c in TARGETS:
        vals = pd.to_numeric(y[c], errors="coerce")
        if vals.isna().any() or not set(vals.unique()).issubset({0,1}):
            raise ValueError(f"Target {c} must contain only binary 0/1 values without missing data.")
        y[c] = vals.astype(int)
    return X, y

def build_pipeline(X, random_state=42):
    numeric = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    preprocess = ColumnTransformer([
      ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
      ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ], remainder="drop")
    base = RandomForestClassifier(n_estimators=250, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=random_state, n_jobs=-1)
    return Pipeline([("preprocess", preprocess), ("model", MultiOutputClassifier(base, n_jobs=1))])

def positive_probability(proba, classes):
    positions = np.where(np.asarray(classes) == 1)[0]
    return np.asarray(proba)[:, int(positions[0])] if len(positions) else np.zeros(len(proba))

def train_and_evaluate(df, random_state=42):
    X, y = split_features_targets(df)
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.20, random_state=random_state)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    pipeline = build_pipeline(X_train, random_state)
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    probas = pipeline.predict_proba(X_test)
    metrics=[]
    estimators=pipeline.named_steps["model"].estimators_
    for i,t in enumerate(TARGETS):
        yt=y_test[t].to_numpy(); yp=pred[:,i].astype(int)
        p1=positive_probability(probas[i], estimators[i].classes_)
        auc=float(roc_auc_score(yt,p1)) if len(np.unique(yt))>1 else float("nan")
        metrics.append({"outcome":DISPLAY_NAMES[t],"target_column":t,"accuracy":float(accuracy_score(yt,yp)),
          "precision":float(precision_score(yt,yp,zero_division=0)),"recall":float(recall_score(yt,yp,zero_division=0)),
          "f1":float(f1_score(yt,yp,zero_division=0)),"roc_auc":auc,"test_support":int(len(yt))})
    return {"pipeline":pipeline,"feature_columns":X.columns.tolist(),"target_columns":TARGETS,
      "train_indices":train_idx,"test_indices":test_idx,"metrics":metrics,
      "test_actual":y_test.reset_index(drop=True),"test_features":X_test.reset_index(drop=True),
      "test_original_indices":df.index.to_numpy()[test_idx],"dataset_rows":len(df),"dataset_columns":df.shape[1]}

def predict_patient(bundle, patient):
    X=patient[bundle["feature_columns"]].copy(); pipeline=bundle["pipeline"]
    pred=pipeline.predict(X)[0]; probas=pipeline.predict_proba(X)
    estimators=pipeline.named_steps["model"].estimators_; rows=[]
    for i,t in enumerate(TARGETS):
        p=float(positive_probability(probas[i],estimators[i].classes_)[0])
        rows.append({"Organ / complication":DISPLAY_NAMES[t],"Predicted class":"Risk detected" if int(pred[i])==1 else "No risk detected","Risk probability (%)":round(100*p,1)})
    return pd.DataFrame(rows)

def save_bundle(bundle,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True); joblib.dump(bundle,path)
def load_bundle(path): return joblib.load(path)
