# Sentiment Analysis MLOps

An end-to-end **Sentiment Analysis MLOps project** that demonstrates the complete machine learning lifecycle, from data ingestion and preprocessing to model training, experiment tracking, API serving, containerization, deployment, CI/CD, and monitoring.

The project uses the **IMDb Movie Reviews dataset** and compares multiple machine learning models to select the best-performing sentiment classifier.

---

## 🚀 Project Overview

The pipeline performs the following operations:

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
      Model Serialization
            │
            ▼
         FastAPI
            │
            ▼
          Docker
            │
            ▼
     Prometheus + Grafana



```

✨ Features

IMDb sentiment classification
Automated data ingestion
Text preprocessing
Feature engineering
Multiple machine learning models
Model evaluation using F1-score
Automatic best-model selection
MLflow experiment tracking
ZenML pipeline orchestration
FastAPI prediction API
Docker containerization
Render deployment configuration
Prometheus monitoring
Grafana dashboard
GitHub Actions CI/CD
Model serialization using Joblib

🤖 Machine Learning Models

The current pipeline compares:

Model	Type
Naive Bayes	Machine Learning
Random Forest	Machine Learning
Linear SVM	Machine Learning

The models are evaluated using their F1-scores, and the model with the highest F1-score is automatically selected as the best model.

📊 Evaluation Metrics

The main model-selection metric is:

F1-Score

The project can also evaluate:

Accuracy
Precision
Recall
F1-Score

F1-score is used as the primary metric because it provides a balance between precision and recall for sentiment classification.

🛠️ Tech Stack

Technology	Purpose
Python	Programming language
Scikit-learn	Machine learning
Pandas	Data processing
NumPy	Numerical computation
ZenML	MLOps pipeline orchestration
MLflow	Experiment tracking and model artifacts
FastAPI	Model serving
Docker	Containerization
Render	Deployment
Prometheus	Application monitoring
Grafana	Metrics visualization
GitHub Actions	CI/CD
Joblib	Model serialization


📁 Project Structure
```
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

```

⚙️ Installation


1. Clone the Repository
git clone https://github.com/AarjavShahh/Sentiment-Analysis-ML_Ops.git
cd Sentiment-Analysis-ML_Ops
2. Create a Virtual Environment
python3.12 -m venv venv
3. Activate the Virtual Environment

For Linux/WSL:

source venv/bin/activate

For Windows:

venv\Scripts\activate
4. Install Dependencies
pip install -r requirements.txt

🔄 Running the MLOps Pipeline

From the project root:

export PYTHONPATH=$PWD:$PYTHONPATH
python -m pipelines.sentiment_pipeline

The ZenML pipeline performs:

Data ingestion
Data preprocessing
Feature engineering
Model training
Model evaluation
Model comparison
Best-model selection
Model serialization
MLflow tracking

The final model is saved as:

models/best_model.joblib

📈 MLflow

MLflow is used for experiment tracking and artifact management.

Start the MLflow server:

mlflow server --host 0.0.0.0 --port 5000

Open the MLflow dashboard:

http://127.0.0.1:5000

MLflow tracks:

Model parameters
Model metrics
F1-score
Best model
Model artifacts
Training runs

🧪 ZenML

ZenML is used to organize the machine learning workflow into reproducible pipeline steps.

The main pipeline is:

pipelines/sentiment_pipeline.py

Individual pipeline steps are located inside:

steps/

Run the pipeline using:

python -m pipelines.sentiment_pipeline
🌐 FastAPI

The trained model is served through FastAPI.

Start the API:

uvicorn api.app:app --host 0.0.0.0 --port 8000

Open the API:

http://127.0.0.1:8000

Open the interactive Swagger documentation:

http://127.0.0.1:8000/docs
Example Request
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

Access the API:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs
☁️ Render Deployment

The project contains:

render.yaml

which provides the configuration required for deployment on Render.

Deployment flow:

GitHub
   │
   ▼
Docker Container
   │
   ▼
FastAPI
   │
   ▼
Sentiment Prediction
📊 Prometheus Monitoring

Prometheus is used to collect application metrics.

Configuration:

monitoring/prometheus.yml

Start Prometheus:

prometheus --config.file=monitoring/prometheus.yml

Open the Prometheus dashboard:

http://127.0.0.1:9090
📉 Grafana Dashboard

Grafana is used to visualize the metrics collected by Prometheus.

Start Grafana:

grafana-server

Open Grafana:

http://127.0.0.1:3000

The project contains a dashboard configuration:

grafana/dashboard.json

The dashboard can be imported into Grafana to visualize application metrics.

🔁 CI/CD

GitHub Actions is configured through:

.github/workflows/ci.yml

The CI workflow can automatically run project checks whenever code is pushed to GitHub.

Workflow:

Developer
    │
    ▼
Git Commit
    │
    ▼
GitHub Push
    │
    ▼
GitHub Actions
    │
    ├── Install Dependencies
    ├── Run Tests
    └── Validate Project


    ```
🔬 Complete MLOps Architecture
                    ┌─────────────────────┐
                    │   IMDb Dataset      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Data Ingestion     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Preprocessing       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
          ┌──────────┐   ┌───────────┐   ┌──────────┐
          │  Naive   │   │   Random  │   │  Linear  │
          │  Bayes   │   │   Forest  │   │   SVM    │
          └────┬─────┘   └─────┬─────┘   └────┬─────┘
               │               │              │
               └───────────────┼──────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Evaluation    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Best Model          │
                    │ Selection           │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
          ┌─────────────┐             ┌─────────────┐
          │   MLflow    │             │ Best Model  │
          │  Tracking   │             │   Joblib    │
          └─────────────┘             └──────┬──────┘
                                             │
                                             ▼
                                      ┌─────────────┐
                                      │   FastAPI   │
                                      └──────┬──────┘
                                             │
                                             ▼
                                      ┌─────────────┐
                                      │    Docker   │
                                      └──────┬──────┘
                                             │
                                       


                         Monitoring
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
         ┌─────────────┐             ┌─────────────┐
         │ Prometheus  │────────────►│   Grafana   │
         └─────────────┘             └─────────────┘
```

🔐 Security

Do not commit sensitive information such as:

.env
API keys
Access tokens
Passwords
Cloud credentials
Private keys

Use .gitignore to prevent sensitive and unnecessary files from being committed.

🧪 Testing

Run the test suite:

pytest

Check Python files for syntax errors:

python -m compileall api pipelines steps


🚀 Local Development Workflow

Terminal 1 — MLflow
mlflow server --host 0.0.0.0 --port 5000
Terminal 2 — ZenML Pipeline
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
BiLSTM deep learning model
Optuna hyperparameter optimization
ML + DL model comparison
Automated model retraining
Model versioning
Hugging Face Hub integration
Advanced API monitoring
Automated model deployment
Improved Grafana dashboards
Additional automated tests
Data and model drift detection


👨‍💻 Author

Aarjav Shahh

MSc Data Science

GitHub Repository:

https://github.com/AarjavShahh/Sentiment-Analysis-ML_Ops
