from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from src.pipeline import read_dataset, train_and_evaluate, save_bundle, load_bundle, TARGETS, DISPLAY_NAMES
from src.network_analysis import feature_associations, disease_associations, build_graph

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "realistic_multi_organ_diabetes_dataset_4000.csv"
MODEL_PATH = ROOT / "artifacts" / "risk_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "evaluation_metrics.json"
FRONTEND_PATH = ROOT / "frontend" / "index.html"

st.set_page_config(page_title="SixPath | Clinical Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],footer{display:none!important}
[data-testid="stAppViewContainer"]{background:#07111f}
.block-container{padding:0!important;max-width:100%!important}
[data-testid="stMainBlockContainer"]{padding:0!important}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    return read_dataset(DATA_PATH)

@st.cache_resource(show_spinner="Preparing the clinical intelligence workspace…")
def get_bundle(signature):
    current = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if MODEL_PATH.exists():
        try:
            saved = load_bundle(MODEL_PATH)
            if saved.get("data_signature") == current:
                return saved
        except Exception:
            pass
    df = load_data()
    bundle = train_and_evaluate(df, random_state=42)
    bundle["data_signature"] = current
    save_bundle(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(bundle["metrics"], indent=2), encoding="utf-8")
    return bundle

@st.cache_data(show_spinner="Building the feature–disease relationship map…")
def get_network_payload():
    df = load_data()
    features = [c for c in df.columns if c not in TARGETS and c.strip().lower() not in {"patient_id", "record_id", "id", "index"}]
    fa = feature_associations(df, features)
    da = disease_associations(df)
    graph = build_graph(fa, da, threshold=0.16, top_n=6)
    # Deterministic layout is used only to give the interactive canvas a clean initial arrangement.
    pos = nx.spring_layout(graph, seed=19, k=0.75, iterations=90, weight="weight")
    nodes = list(graph.nodes(data=True))
    index = {node: i for i, (node, _) in enumerate(nodes)}
    payload_nodes = []
    for node, attrs in nodes:
        x, y = pos[node]
        payload_nodes.append({
            "id": str(node), "label": str(attrs.get("label", node)),
            "type": str(attrs.get("node_type", "feature")),
            "x": float((x + 1.15) / 2.3), "y": float((y + 1.15) / 2.3),
            "note": str(attrs.get("note", ""))
        })
    payload_edges = []
    for a, b, attrs in graph.edges(data=True):
        payload_edges.append({
            "a": index[a], "b": index[b],
            "weight": float(attrs.get("weight", 0.1)),
            "kind": "curated" if "curated" in str(attrs.get("kind", "")).lower() else "association",
            "note": str(attrs.get("biological_basis", attrs.get("kind", "Dataset association")))
        })
    return {"nodes": payload_nodes, "edges": payload_edges}


def clean_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)

try:
    df = load_data()
    signature = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    bundle = get_bundle(signature)
    network = get_network_payload()
except Exception as exc:
    st.error(f"Unable to prepare the workspace: {exc}")
    st.stop()

# Precompute only the held-out test split. The UI never presents training-set predictions as evaluation.
test_x = bundle["test_features"].copy()
proba = bundle["pipeline"].predict_proba(test_x)
estimators = bundle["pipeline"].named_steps["model"].estimators_
original_indices = np.asarray(bundle["test_original_indices"], dtype=int)
records = []
for row_i in range(len(test_x)):
    risks = []
    for out_i, target in enumerate(TARGETS):
        classes = np.asarray(estimators[out_i].classes_)
        pos = np.where(classes == 1)[0]
        probability = float(proba[out_i][row_i, int(pos[0])]) if len(pos) else 0.0
        risks.append(round(probability * 100, 1))
    features = {str(k): clean_value(v) for k, v in test_x.iloc[row_i].to_dict().items()}
    records.append({"id": int(original_indices[row_i]) + 1, "risks": risks, "features": features})

metrics = []
for row in bundle["metrics"]:
    metrics.append({
        "outcome": str(row.get("outcome", "Outcome")),
        "accuracy": clean_value(row.get("accuracy")),
        "precision": clean_value(row.get("precision")),
        "recall": clean_value(row.get("recall")),
        "f1": clean_value(row.get("f1")),
        "roc_auc": clean_value(row.get("roc_auc")),
        "test_support": clean_value(row.get("test_support"))
    })

risk_names = [DISPLAY_NAMES[t] for t in TARGETS]
profile_keys = []
for key in ["HbA1c_percent", "Fasting_Glucose_mg_dL", "Systolic_BP", "Diastolic_BP", "BMI", "Serum_Creatinine_mg_dL", "eGFR_mL_min", "Urine_Albumin_mg_g", "Total_Cholesterol_mg_dL", "Triglycerides_mg_dL"]:
    if key in test_x.columns:
        profile_keys.append(key)
profile_keys.extend([c for c in test_x.columns if c not in profile_keys][:max(0, 10-len(profile_keys))])

app_data = {
    "datasetRows": int(len(df)),
    "featureCount": int(len(bundle["feature_columns"])),
    "records": records,
    "metrics": metrics,
    "riskNames": risk_names,
    "profileKeys": profile_keys,
    "network": network
}
serialized = json.dumps(app_data, ensure_ascii=False, separators=(",", ":"), default=clean_value).replace("</", "<\\/")
html = FRONTEND_PATH.read_text(encoding="utf-8").replace("{{APP_DATA}}", serialized)
components.html(html, height=1050, scrolling=True)
