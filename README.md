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
