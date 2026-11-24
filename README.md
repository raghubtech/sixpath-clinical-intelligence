# Multi-Organ Clinical Risk Intelligence — SixPath product preview

A product-style clinical analytics workspace for exploring six diabetes-related complication domains, patient-level model estimates, held-out evaluation and an interlinked feature–disease network.

The interface is presented as **SixPath Clinical Intelligence**, a working product concept name for this portfolio demonstration (not a cleared or registered commercial brand). The frontend is built with HTML, CSS and browser JavaScript; Streamlit serves the Python model and embeds the frontend. It includes a dark clinical theme, animated vascular-flow background that responds to scrolling, interactive patient selection, risk dashboards, evaluation views and a hoverable relationship network.

## Project structure

```text
app.py                       Python/Streamlit host and model-data bridge
frontend/index.html           HTML + CSS + JavaScript product interface
train_model.py               training/evaluation entrypoint
launch.ps1                   setup + local launch
setup_and_publish.ps1         setup + train + GitHub push
requirements.txt
src/pipeline.py               preprocessing, multi-output model, metrics, prediction
src/network_analysis.py        dataset associations + curated biological context
data/                         4,000-row datasets
artifacts/risk_model.joblib    trained model bundle
docs/project_report_5000_words.docx
notebooks/legacy_colab_pipeline_reference.ipynb
```

## Product experience

- Product landing hero and navigation
- Animated blood-vessel-inspired flow paths with moving particles; flow intensity responds to scroll position
- Patient selector with six complication probability bars
- Clinical feature snapshot for the selected held-out record
- Per-outcome F1 and ROC-AUC views
- Interactive feature–disease network with hover highlighting
- Separate visual evidence layers for dataset association and curated biological context
- Responsive layout for desktop and mobile

The frontend is implemented in plain HTML/CSS/JavaScript; the model and data pipeline remain Python. This avoids adding a separate Java runtime or frontend build toolchain.

## Dataset

The primary CSV has 4,000 rows and 37 columns (31 input columns + six binary outcomes). It is synthetic, not real patient data. The model predicts:

- Nephropathy
- Cardiovascular disease
- Hepatic complications
- Neuropathy
- Retinopathy
- Cerebrovascular disease

The six outcome columns are excluded from input features. The patient selector only displays held-out test records that were not used to fit the model.

## Run from Windows PowerShell

Install Python 3.11 and Git for Windows. Extract the ZIP, open PowerShell in the extracted project folder, then run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\launch.ps1
```

Keep the terminal open. The local app URL is normally `http://localhost:8501`.

## Push to GitHub and deploy

1. Create a new **empty** GitHub repository (do not initialize it with README/license).
2. Open PowerShell in the extracted project folder.
3. Run:

```powershell
.\setup_and_publish.ps1 -OpenStreamlitCloud
```

Paste the empty repository's HTTPS URL when prompted. The script installs dependencies, trains/evaluates, commits and pushes `main`, then opens Streamlit Community Cloud. Sign in through GitHub and select repository, branch `main`, app file `app.py`, then click Deploy.

GitHub authentication is handled by Git Credential Manager. Never paste a GitHub password or token into ChatGPT or into source code.

## Scientific interpretation and limitations

Dataset-derived edges are statistical associations. Curated biological edges are contextual mechanism notes; neither layer proves direct causation or temporal disease progression. Synthetic target labels can be rule-derived, which may produce near-perfect scores. Those scores are not clinical validation.

The deployed baseline is a multi-output Random Forest with one binary estimator per complication and a shared preprocessing pipeline. It is a research prototype, not a medical device. Before any clinical use, retrain and validate with appropriately governed real data, external validation, calibration, subgroup analysis and clinical review.

**Important:** The dashboard is for education, portfolio demonstration and research exploration only. It must not be used for diagnosis, treatment or clinical decisions.

## Troubleshooting

- `setup_and_publish.ps1 is not recognized`: PowerShell is not in the extracted project directory. Run `cd` into the folder containing the script, or open PowerShell from File Explorer's address bar while inside that folder.
- Python not found: install Python 3.11, enable Add Python to PATH, then restart PowerShell.
- Git not found: install Git for Windows and restart PowerShell.
- Script blocked: use the process-scoped execution-policy command above.
- Push rejected: ensure the remote is empty and authenticate with Git Credential Manager.
- Missing outcome columns: use the primary CSV or provide all six expected target columns.
