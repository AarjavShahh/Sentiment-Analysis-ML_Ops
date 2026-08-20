# 🎬 Sentiment Analysis MLOps Pipeline

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![ZenML](https://img.shields.io/badge/Orchestration-ZenML-purple.svg)](https://zenml.io/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Container-Docker-blue.svg)](https://www.docker.com/)
[![Prometheus](https://img.shields.io/badge/Monitoring-Prometheus-orange.svg)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Dashboard-Grafana-orange.svg)](https://grafana.com/)

An end-to-end **Sentiment Analysis MLOps project** demonstrating the complete machine learning lifecycle: data ingestion, preprocessing, model training, experiment tracking, API serving, containerization, deployment, CI/CD, and monitoring.

The project uses the **IMDb Movie Reviews dataset** and compares multiple machine learning models to select and deploy the best sentiment classifier.

---

## 🚀 Lifecycle Workflow

```text
IMDb Movie Reviews Dataset
            │
            ▼
      Data Ingestion
            │
            ▼
     Data Preprocessing
            │
            ▼
     Feature Engineering
            │
            ▼
    ┌───────┼────────┐
    ▼       ▼        ▼
 Naive   Random    Linear
 Bayes   Forest      SVM
    │       │        │
    └───────┼────────┘
            │
            ▼
      Model Evaluation
            │
            ▼
     Best Model Selection & Serialization (Joblib)
            │
            ▼
    ┌───────┴───────┐
    ▼               ▼
 FastAPI Serving   MLflow Experiment
  (Port 8000)      Tracking (Port 5000)
    │               │
    ▼               ▼
 Prometheus      Grafana Dashboards
  (Port 9090)     (Port 3000)
```

---

## 🛠️ Tech Stack & Architecture

| Component | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.12 | Core programming environment |
| **ML Framework** | Scikit-learn, Pandas, NumPy | Data ingestion, vectorization, & modeling |
| **Orchestration** | ZenML | Pipelines DAG orchestration |
| **Tracking** | MLflow | Parameter, metrics, & artifact logging |
| **Serving** | FastAPI & Uvicorn | High-performance inference endpoints |
| **Monitoring** | Prometheus | Real-time service metrics exporter & scraper |
| **Visualization** | Grafana | System & performance dashboard |
| **Containerization**| Docker / Docker Compose | Service encapsulation & portability |
| **CI/CD** | GitHub Actions | Lint checks, syntax validation, & test runner |

---

## 📁 Repository Structure

```text
├── .github/workflows/   # CI/CD pipelines (GitHub Actions)
├── .zen/                # ZenML local repository settings
├── api/
│   ├── __init__.py
│   └── app.py           # FastAPI server with prometheus instrumentator
├── grafana/
│   └── dashboard.json   # Pre-configured Grafana metrics panel
├── mlflow/
│   ├── artifacts/       # Local MLflow run files and binaries
│   └── mlflow.db        # Backend sqlite metadata database
├── models/
│   └── best_model.joblib# Serialized best model and pipeline vectorizer
├── monitoring/
│   └── prometheus.yml   # Prometheus scrape specifications & targets
├── pipelines/
│   └── sentiment_pipeline.py # Orchestrated ZenML ML pipeline definition
├── steps/
│   ├── data_ingestion.py
│   ├── preprocess_data.py
│   ├── feature_engineering.py
│   ├── train_naive_bayes.py
│   ├── train_random_forest.py
│   ├── train_svm.py
│   ├── select_best_model.py
│   └── save_best_model.py
├── Dockerfile           # Minimal API production server Dockerfile
├── docker-compose.yml   # Infrastructure composed setup (MLflow, Prom, Grafana, API)
├── requirements.txt     # Python dependency definition file
└── README.md            # Project description & guide
```

---

## ⚙️ Quick Start Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AarjavShahh/Sentiment-Analysis-ML_Ops.git
cd Sentiment-Analysis-ML_Ops
```

### 2. Configure Virtual Environment

Create and activate a virtual environment (Python 3.12 recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS / WSL
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔄 Running the MLOps Pipeline

Run the ZenML orchestrated pipeline to ingest, clean, evaluate, track, and output the best-performing model to `models/best_model.joblib`:

```bash
# Ensure PYTHONPATH includes root
export PYTHONPATH=$PWD:$PYTHONPATH  # Linux/WSL
# or PC PowerShell:
$env:PYTHONPATH="."

python -m pipelines.sentiment_pipeline
```

---

## 🌐 Serving & Dashboards URLs

All local and containerized services are mapped to their respective local loops:

| Service | Address | Description |
|---|---|---|
| **FastAPI Swagger UI** | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) | Interactive API testing documentation |
| **ZenML Dashboard** | [http://127.0.0.1:8237/](http://127.0.0.1:8237/) | Pipeline execution logs & DAG visualizations |
| **MLflow Tracking UI** | [http://127.0.0.1:5000/](http://127.0.0.1:5000/) | Metrics, params & model registry tracker |
| **Prometheus Server** | [http://127.0.0.1:9090/](http://127.0.0.1:9090/) | Scraping state & time-series database UI |
| **Grafana Dashboard** | [http://127.0.0.1:3000/](http://127.0.0.1:3000/) | System metrics & sentiment request visualizations |

---

## 🚀 Serving and Infrastructure Setup

### Starting the Serving API (Host Mode)
Launch the FastAPI application on your local machine using `uvicorn`:
```bash
uvicorn api.app:app --host 127.0.0.1 --port 8000
```
#### Example Request
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "This movie was absolutely amazing! The story, acting, and direction were excellent."
}'
```
#### Output Payload
```json
{
  "text": "This movie was absolutely amazing! The story, acting, and direction were excellent.",
  "prediction": 1,
  "sentiment": "positive",
  "model": "Linear SVM"
}
```

### Starting Infrastructure (Containers)
To launch MLflow, Prometheus, and Grafana in the background, spin up Docker Compose:
```bash
docker compose up -d
```
This launches:
- **MLflow Tracking** connected to a SQLite DB (`mlflow/mlflow.db`).
- **Prometheus** configured via `monitoring/prometheus.yml` (scraping host FastAPI metrics through `host.docker.internal:8000`).
- **Grafana** loaded and connected to Prometheus.

---

## 📈 Monitoring Configuration

### 1. Prometheus Target Scrapes
Prometheus pulls real-time tracking metrics from Python's FastAPI process. The target is defined in `monitoring/prometheus.yml`:
```yaml
scrape_configs:
  - job_name: "fastapi-app"
    scrape_interval: 5s
    static_configs:
      - targets: ["host.docker.internal:8000"]
```

### 2. Grafana Dashboard Import
To visualize metrics:
1. Open Grafana at [http://localhost:3000/](http://localhost:3000/).
2. Add Prometheus (`http://prometheus:9090`) as a Data Source.
3. Import the pre-configured layout from `grafana/dashboard.json` to monitor server request rate, errors, latencies, and sentiment distributions.

---

## 🧪 Testing and Quality Control

Ensure syntax coherence and run tests:
```bash
# Run Pytest suite
pytest

# Compile assets syntax check
python -m compileall api pipelines steps
```
