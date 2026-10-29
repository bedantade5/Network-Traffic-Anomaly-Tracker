# Network Traffic Anomaly & Data Exfiltration Tracker

## Project Overview

This project implements an end-to-end network traffic anomaly detection pipeline using the **UNSW-NB15** dataset. It trains an **Isolation Forest** (unsupervised anomaly detector) on flow-level network features, scores a held-out test split, and evaluates results against ground-truth attack labels. An interactive **Streamlit dashboard** allows filtering, visualization, and forensic review of flagged connections.

The pipeline is structured into modular stages — ingestion, feature engineering, model training, scoring, evaluation — with all results persisted for reproducibility and inspection.

## Novelty of the Project

This implementation combines several techniques that are individually established but rarely integrated into a single, reproducible, and interactive pipeline:

1. **Behavioral feature engineering.** Beyond the dataset's raw flow statistics, the pipeline computes derived features that capture data-exfiltration-relevant behavior: `bytes_per_second` (transfer rate), `outbound_fraction` (ratio of source bytes to total), `inbound_outbound_ratio`, `packet_ratio`, and per-packet byte averages. These are defined in [`src/engineering.py`](src/engineering.py) using safe division to avoid infinity values.

2. **Unsupervised anomaly detection.** The Isolation Forest is trained on flow features alone — ground-truth labels (`label`, `attack_cat`) are never included in the feature matrix. This makes the approach applicable to real-world networks where labeled data is unavailable. An optional `--normal-only` mode allows a label-assisted comparison experiment.

3. **Continuous anomaly ranking.** Instead of binary classification, each connection receives a continuous `anomaly_score` (negated decision function), enabling ranked triage. Analysts can review the most suspicious connections first rather than treating all flagged flows equally.

4. **Ground-truth evaluation without label leakage.** When the test split includes `label` and `attack_cat` columns, the pipeline automatically computes precision, recall, F1, confusion matrix, and **per-category attack recall** (e.g., DoS, Exploits, Worms) — all without the labels ever influencing the model.

5. **Interactive forensic dashboard.** The Streamlit application (`app.py`) provides filtering by status, attack category, and service; a scatter plot of transfer rates; a ranked table of suspicious connections; and full evaluation metrics — allowing an analyst to interactively investigate flagged traffic.

## Dataset Source

This project uses the **UNSW-NB15** dataset, a network intrusion detection benchmark created by the Australian Centre for Cyber Security (ACCS) at UNSW Canberra.

- **Dataset:** UNSW-NB15 Training and Testing Sets (CSV)
- **Files used:**
  - `UNSW_NB15_training-set.csv` (175,341 rows)
  - `UNSW_NB15_testing-set.csv` (82,332 rows)
- **Source:** [https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- **Citation:**
  > Moustafa, N. & Slay, J. (2015). UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems. *Military Communications and Information Systems Conference (MilCIS)*, IEEE.

Place both CSV files in the `data/` directory before running the pipeline.

## Requirements

- **Python:** 3.10 or newer
- **Dependencies** (from [`requirements.txt`](requirements.txt)):

  | Package | Version |
  |---|---|
  | numpy | ≥1.24, <3 |
  | pandas | ≥2.0, <3 |
  | pyarrow | ≥14 |
  | scikit-learn | ≥1.3, <2 |
  | joblib | ≥1.3 |
  | streamlit | ≥1.35 |
  | plotly | ≥5.20 |

## Installation and Execution

From the project root directory:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# macOS / Linux:
# source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the anomaly detection pipeline
python main.py --train data/UNSW_NB15_training-set.csv --test data/UNSW_NB15_testing-set.csv

# 4. Launch the interactive dashboard
streamlit run app.py
```

The pipeline prints the evaluation results as JSON to the console and writes all outputs to the `outputs/` directory.

## Reproducing the Main Result

To reproduce the anomaly detection results from scratch:

**Step 1 — Ensure the dataset files are present:**

```
data/UNSW_NB15_training-set.csv
data/UNSW_NB15_testing-set.csv
```

**Step 2 — Run the pipeline:**

```bash
python main.py --train data/UNSW_NB15_training-set.csv --test data/UNSW_NB15_testing-set.csv
```

This trains an Isolation Forest (`contamination=0.01`, `random_state=42`, `n_estimators=100`) on the full training split and scores all 82,332 test rows. The process is deterministic.

**Step 3 — Verify the outputs:**

The following files are generated in `outputs/`:

| File | Description |
|---|---|
| `processed_scored_logs.parquet` | All test rows with engineered features, `anomaly_prediction`, and `anomaly_score` |
| `metrics.json` | Cleaning statistics, selected features, flagged count, and held-out evaluation metrics |
| `model.joblib` | Fitted scikit-learn pipeline (preprocessing + Isolation Forest) |

**Step 4 — View results in the dashboard:**

```bash
streamlit run app.py
```

The dashboard reads from `outputs/` and displays KPI metrics, the anomaly scatter plot, the ranked suspicious connections table, precision/recall/F1, confusion matrix, and per-category detection rates.

## Results and Outputs

With the default settings (`contamination=0.01`, full training split), the pipeline produces the following evaluation on the held-out test split (82,332 rows):

| Metric | Value |
|---|---|
| Flagged flows | 668 |
| Precision | 0.516 |
| Recall | 0.008 |
| F1 Score | 0.015 |
| True Positives | 345 |
| False Positives | 323 |
| True Negatives | 36,677 |
| False Negatives | 44,987 |

Per-category recall is available in `outputs/metrics.json` and on the dashboard, covering all 10 UNSW-NB15 categories (Normal, Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms).

## Project Structure

```
Network_Anomaly_Tracker/
├── app.py                 # Streamlit dashboard
├── main.py                # Pipeline entry point (train, score, evaluate)
├── requirements.txt       # Python dependencies
├── README.md
├── data/                  # Dataset files (user-supplied)
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
├── outputs/               # Generated results
│   ├── processed_scored_logs.parquet
│   ├── metrics.json
│   └── model.joblib
└── src/                   # Pipeline modules
    ├── __init__.py
    ├── ingestion.py       # Data loading and validation
    ├── engineering.py     # Feature engineering
    ├── model.py           # Isolation Forest pipeline
    └── evaluation.py      # Held-out evaluation metrics
```

## Limitations

- **Anomalies are not confirmed exfiltration.** Flagged connections are statistically unusual flows, not verified data breaches. The UNSW-NB15 dataset does not provide a dedicated exfiltration ground truth.
- **Limited forensic context.** The processed UNSW-NB15 training and testing splits do not include usable IP addresses or timestamps, limiting post-detection forensic analysis.
- **Low recall is expected.** The Isolation Forest with `contamination=0.01` prioritizes precision over recall. It flags volumetric and behavioral outliers, which overlap with only a subset of the dataset's labeled attack categories.
