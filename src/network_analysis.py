"""Statistical association graph plus a separately identified curated biology layer."""
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import pointbiserialr, chi2_contingency
from .pipeline import TARGETS, DISPLAY_NAMES

BIOLOGICAL_NOTES={
"Nephropathy_Risk":"Kidney microvascular injury; filtration and albumin leakage are clinically relevant.",
"Cardiovascular_Risk":"Endothelial dysfunction, atherosclerosis, blood pressure and metabolic risk.",
"Hepatic_Risk":"Insulin resistance and dysregulated lipid metabolism are linked to metabolic fatty liver.",
"Neuropathy_Risk":"Hyperglycemia, oxidative stress and microvascular nerve injury.",
"Retinopathy_Risk":"Chronic hyperglycemia and retinal microvascular injury.",
"Cerebrovascular_Risk":"Vascular injury, hypertension and atherosclerosis affect cerebral circulation."}
# Each entry maps feature-name tokens to outcomes with a concise shared-mechanism note.
FEATURE_RULES=[
(("hba1c","fasting_glucose","postprandial_glucose"),["Nephropathy_Risk","Retinopathy_Risk","Neuropathy_Risk","Cardiovascular_Risk","Cerebrovascular_Risk"],"Chronic hyperglycemia is associated with metabolic and microvascular injury."),
(("creatinine","egfr","uacr","albumin_creatinine","urine_albumin","blood_urea","bun"),["Nephropathy_Risk","Cardiovascular_Risk"],"Kidney function and albuminuria are relevant to kidney damage and cardiorenal pathways."),
(("systolic_bp","diastolic_bp","blood_pressure"),["Nephropathy_Risk","Cardiovascular_Risk","Cerebrovascular_Risk","Retinopathy_Risk"],"Hypertension affects renal pressure and vascular risk, including stroke and retinal injury."),
(("bmi","waist","homa_ir","insulin_level"),["Hepatic_Risk","Cardiovascular_Risk"],"Adiposity and insulin resistance are linked to metabolic liver and cardiovascular disease."),
(("triglyceride","ldl","cholesterol"),["Cardiovascular_Risk","Cerebrovascular_Risk","Hepatic_Risk"],"Atherogenic lipids relate to vascular disease; lipid dysregulation also accompanies metabolic liver disease."),
(("vibration","monofilament","nerve_conduction"),["Neuropathy_Risk"],"These measures assess peripheral sensory or nerve function."),
(("retinal","visual_acuity"),["Retinopathy_Risk"],"Retinal structure and visual acuity are ophthalmic measures relevant to retinopathy.")]
DISEASE_RULES=[
("Nephropathy_Risk","Cardiovascular_Risk","Cardiorenal interaction: shared vascular, blood-pressure and metabolic pathways."),
("Nephropathy_Risk","Retinopathy_Risk","Shared diabetic microvascular injury may affect kidney glomeruli and retinal capillaries."),
("Neuropathy_Risk","Retinopathy_Risk","These are diabetic microvascular complications with shared hyperglycemic injury mechanisms."),
("Cardiovascular_Risk","Cerebrovascular_Risk","Atherosclerosis and endothelial dysfunction can affect coronary and cerebral vessels."),
("Hepatic_Risk","Cardiovascular_Risk","Insulin resistance and metabolic dysfunction connect metabolic liver and cardiovascular risk.")]

def cramers_v(x,y):
    tab=pd.crosstab(x.astype(str),y.astype(str))
    if tab.shape[0]<2 or tab.shape[1]<2:return 0.0
    chi=chi2_contingency(tab,correction=False)[0]; n=tab.to_numpy().sum(); den=n*min(tab.shape[0]-1,tab.shape[1]-1)
    return float(np.sqrt(chi/den)) if den>0 else 0.0

def feature_associations(df,features):
    rows=[]
    for f in features:
        s=df[f]; n=pd.to_numeric(s,errors="coerce"); numeric=n.notna().sum()>=max(3,int(.8*s.notna().sum()))
        for t in TARGETS:
            y=pd.to_numeric(df[t],errors="coerce"); valid=y.notna()&s.notna()
            if valid.sum()<10 or y[valid].nunique()<2: score=signed=0.; method="insufficient variation"
            elif numeric:
                try: signed=float(pointbiserialr(y[valid].astype(int),n[valid].astype(float)).statistic)
                except Exception: signed=0.
                if not np.isfinite(signed):signed=0.
                score=abs(signed); method="point-biserial r"
            else:
                score=cramers_v(s[valid],y[valid].astype(int)); signed=score; method="Cramer's V (unsigned)"
            rows.append({"feature":f,"target":t,"disease":DISPLAY_NAMES[t],"score":float(score),"signed_association":float(signed),"method":method})
    return pd.DataFrame(rows)

def disease_associations(df):
    rows=[]
    for i,a in enumerate(TARGETS):
      for b in TARGETS[i+1:]:
        x=pd.to_numeric(df[a],errors="coerce"); y=pd.to_numeric(df[b],errors="coerce"); valid=x.notna()&y.notna()
        r=float(np.corrcoef(x[valid],y[valid])[0,1]) if valid.sum()>10 and x[valid].nunique()>1 and y[valid].nunique()>1 else 0.
        if not np.isfinite(r):r=0.
        rows.append({"disease_a":a,"disease_b":b,"label_a":DISPLAY_NAMES[a],"label_b":DISPLAY_NAMES[b],"correlation":r,"score":abs(r)})
    return pd.DataFrame(rows)

def build_graph(fa,da,threshold=.15,top_n=7):
    G=nx.Graph()
    for t in TARGETS:G.add_node(t,node_type="disease",label=DISPLAY_NAMES[t],note=BIOLOGICAL_NOTES[t])
    for t in TARGETS:
        sub=fa[fa.target==t].sort_values("score",ascending=False)
        for _,r in sub[sub.score>=threshold].head(top_n).iterrows():
            f=str(r.feature);G.add_node(f,node_type="feature",label=f)
            G.add_edge(f,t,weight=float(r.score),signed=float(r.signed_association),kind="dataset association",method=str(r.method))
    for _,r in da.iterrows():
        if r.score>=threshold:G.add_edge(r.disease_a,r.disease_b,weight=float(r.score),signed=float(r.correlation),kind="dataset disease association",method="binary outcome correlation")
    # Add curated mechanism context for feature names actually present in this dataset.
    lookup={str(c).lower():str(c) for c in fa.feature.unique()}
    for tokens,targets,note in FEATURE_RULES:
        matches=[col for key,col in lookup.items() if any(tok in key for tok in tokens)]
        for f in set(matches):
          for t in targets:
            if G.has_edge(f,t):G[f][t]["biological_basis"]=note;G[f][t]["evidence_kind"]="dataset association + curated biology"
            else:G.add_edge(f,t,weight=.55,signed=0.,kind="curated biological mechanism",method="mechanistic context",biological_basis=note,evidence_kind="curated context; not causal proof")
    for a,b,note in DISEASE_RULES:
        if G.has_edge(a,b):G[a][b]["biological_basis"]=note;G[a][b]["evidence_kind"]="dataset association + curated biology"
        else:G.add_edge(a,b,weight=.55,signed=0.,kind="curated biological mechanism",method="mechanistic context",biological_basis=note,evidence_kind="curated context; not causal proof")
    return G
