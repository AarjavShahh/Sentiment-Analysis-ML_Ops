Sentiment Analysis MLOps

An end-to-end Sentiment Analysis MLOps pipeline that combines traditional machine learning, experiment tracking, model selection, API deployment, containerization, CI/CD, and monitoring.

The project uses the IMDb Movie Reviews dataset and compares multiple machine-learning approaches to identify the best-performing sentiment classification model.

🚀 Project Overview

This project demonstrates how a machine-learning model can be taken from data ingestion to deployment and monitoring using modern MLOps tools.

Pipeline
IMDb Dataset
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
┌─────────────────────────────────────┐
│         Model Training              │
│                                     │
│  ┌───────────────┐                  │
│  │ Naive Bayes   │                  │
│  └───────────────┘                  │
│                                     │
│  ┌───────────────┐                  │
│  │ Random Forest │                  │
│  └───────────────┘                  │
│                                     │
│  ┌───────────────┐                  │
│  │ Linear SVM    │                  │
│  └───────────────┘                  │
└─────────────────────────────────────┘
     │
     ▼
Model Evaluation
     │
     ▼
Best Model Selection
     │
     ▼
MLflow Tracking
     │
     ▼
Save Best Model
     │
     ▼
FastAPI
     │
     ▼
Docker
     │
     ▼
Render Deployment
     │
     ▼
Prometheus ─────► Grafana
✨ Features
IMDb sentiment classification
Automated data ingestion
Text preprocessing
Feature engineering
Multiple ML models
Model comparison using F1 score
Automatic best-model selection
MLflow experiment tracking
ZenML pipeline orchestration
FastAPI prediction endpoint
Docker containerization
Render deployment configuration
Prometheus monitoring
Grafana dashboard
GitHub Actions CI pipeline
Reproducible project structure
🤖 Models

The project currently trains and compares:

Model	Type
Naive Bayes	Machine Learning
Random Forest	Machine Learning
Linear SVM	Machine Learning

The models are evaluated and the model with the highest F1 score is selected as the best model.

📊 Evaluation

The primary model-selection metric is F1-score.

The pipeline also tracks model-specific metrics through MLflow.

Typical evaluation metrics include:

Accuracy
Precision
Recall
F1-score

F1-score is particularly useful for evaluating sentiment classification because it balances precision and recall.

🛠️ Tech Stack
Technology	Purpose
Python	Core programming language
Scikit-learn	Machine learning
Pandas	Data processing
NumPy	Numerical operations
ZenML	Pipeline orchestration
MLflow	Experiment tracking and artifacts
FastAPI	Model serving API
Docker	Containerization
Render	Deployment
Prometheus	Metrics collection
Grafana	Monitoring dashboard
GitHub Actions	CI/CD
Joblib	Model serialization
📁 Project Structure
Sentiment-Analysis-ML_Ops/
│
├── api/
│   ├── __init__.py
│   └── app.py
│
├── pipelines/
│   ├── __init__.py
│   └── sentiment_pipeline.py
│
├── steps/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── preprocess_data.py
│   ├── feature_engineering.py
│   ├── train_naive_bayes.py
│   ├── train_random_forest.py
│   ├── train_svm.py
│   ├── select_best_model.py
│   └── save_best_model.py
│
├── models/
│   └── best_model.joblib
│
├── mlflow/
│   ├── artifacts/
│   └── mlflow.db
│
├── monitoring/
│   └── prometheus.yml
│
├── grafana/
│   └── dashboard.json
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .zen/
│   └── config.yaml
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── render.yaml
└── README.md
⚙️ Installation
1. Clone the repository
git clone https://github.com/AarjavShahh/Sentiment-Analysis-ML_Ops.git
cd Sentiment-Analysis-ML_Ops
2. Create a virtual environment
python3.12 -m venv venv
3. Activate the environment

Linux/WSL:

source venv/bin/activate

Windows:

venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt
▶️ Running the Pipeline

From the project root:

export PYTHONPATH=$PWD:$PYTHONPATH
python -m pipelines.sentiment_pipeline

The pipeline performs:

Data ingestion
Data preprocessing
Feature engineering
Model training
Model evaluation
Model comparison
Best-model selection
Model serialization
MLflow logging

The final model is saved under:

models/best_model.joblib
📈 MLflow

Start MLflow:

mlflow server --host 0.0.0.0 --port 5000

Open the MLflow dashboard:

http://127.0.0.1:5000

MLflow is used to track:

Experiments
Model parameters
Evaluation metrics
Best model
Model artifacts
🌐 FastAPI

The trained model can be served through FastAPI.

Start the API:

uvicorn api.app:app --host 0.0.0.0 --port 8000

Open:

http://127.0.0.1:8000

Swagger API documentation:

http://127.0.0.1:8000/docs
Example prediction
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"text":"This movie was absolutely fantastic!"}'

The API returns the predicted sentiment.

🐳 Docker

Build the Docker image:

docker build -t sentiment-analysis-mlops .

Run the container:

docker run -d \
  --name sentiment-api \
  -p 8000:8000 \
  sentiment-analysis-mlops

Test:

http://127.0.0.1:8000/docs


📊 Prometheus

Prometheus configuration is located at:

monitoring/prometheus.yml

Start Prometheus:

prometheus --config.file=monitoring/prometheus.yml

Prometheus dashboard:

http://127.0.0.1:9090

Prometheus collects application metrics exposed by the API.

📉 Grafana

Grafana is used to visualize the application and API metrics collected by Prometheus.

Start Grafana:

grafana-server

Open:

http://127.0.0.1:3000

The project includes a preconfigured dashboard:

grafana/dashboard.json

The dashboard can be imported into Grafana to visualize metrics such as:

API requests
Request rates
Prediction activity
API performance
Application metrics
🔄 CI/CD

GitHub Actions configuration is located at:

.github/workflows/ci.yml

The CI workflow can automatically perform checks whenever changes are pushed to GitHub.

Typical workflow:

Git Push
   │
   ▼
GitHub Actions
   │
   ├── Install dependencies
   ├── Run checks
   └── Validate project
🔬 MLOps Architecture
                ┌───────────────┐
                │ IMDb Dataset  │
                └───────┬───────┘
                        │
                        ▼
              ┌──────────────────┐
              │ Data Processing  │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Feature Engineer │
              └────────┬─────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     Naive Bayes   Random Forest  Linear SVM
          │            │            │
          └────────────┼────────────┘
                       ▼
                Model Evaluation
                       │
                       ▼
                Best Model
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
          MLflow              Model File
             │                   │
             │                   ▼
             │                FastAPI
             │                   │
             │                   ▼
             │                 Docker
             │                   │
             │                   ▼
             │                Render
             │
             ▼
          Experiment
          Tracking
             
Prometheus ───────────► Grafana
🔐 Project Configuration

Environment-specific configuration should be kept outside the source code.

For local development, environment variables can be configured as needed:

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

Do not commit:

.env
venv/
mlruns/
mlartifacts/
credentials
API keys
access tokens
🧪 Testing

Run the test suite using:

pytest

For additional checks:

python -m compileall api pipelines steps
🚀 Complete Local Workflow

A typical local workflow is:

Terminal 1 — MLflow
mlflow server --host 0.0.0.0 --port 5000
Terminal 2 — Pipeline
cd ~/imdb_mlo
source venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH


python -m pipelines.sentiment_pipeline
Terminal 3 — FastAPI
uvicorn api.app:app --host 0.0.0.0 --port 8000
Terminal 4 — Prometheus
prometheus --config.file=monitoring/prometheus.yml
Terminal 5 — Grafana
grafana-server
📌 Future Improvements

Planned improvements include:

BiLSTM deep-learning model integration
Optuna hyperparameter optimization
Comparison of ML and DL models
Automated model retraining
Model versioning
Hugging Face model hosting
Improved API monitoring
Automated deployment
Advanced Grafana dashboards
More comprehensive automated testing
👨‍💻 Author

Aarjav Shahh

MSc Data Science


