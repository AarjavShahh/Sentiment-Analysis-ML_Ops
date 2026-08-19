from pathlib import Path
import pickle

import mlflow
from zenml import step


@step
def save_best_model(
    random_forest_model,
    random_forest_f1: float,
    svm_model,
    svm_f1: float,
    naive_bayes_model,
    naive_bayes_f1: float,
    vectorizer,
) -> str:

    # ---------------------------------------------------------
    # 1. Select best model based on F1 score
    # ---------------------------------------------------------
    models = {
        "Random Forest": {
            "model": random_forest_model,
            "f1": float(random_forest_f1),
        },
        "Linear SVM": {
            "model": svm_model,
            "f1": float(svm_f1),
        },
        "Naive Bayes": {
            "model": naive_bayes_model,
            "f1": float(naive_bayes_f1),
        },
    }

    best_model_name = max(
        models,
        key=lambda name: models[name]["f1"]
    )

    best_model = models[best_model_name]["model"]
    best_f1 = models[best_model_name]["f1"]

    # ---------------------------------------------------------
    # 2. Create models directory
    # ---------------------------------------------------------
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "best_model.pkl"
    vectorizer_path = model_dir / "vectorizer.pkl"

    # ---------------------------------------------------------
    # 3. Save best model
    # ---------------------------------------------------------
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    # ---------------------------------------------------------
    # 4. Save vectorizer
    # ---------------------------------------------------------
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    # ---------------------------------------------------------
    # 5. Log to the ACTIVE ZenML MLflow run
    # ---------------------------------------------------------
    # IMPORTANT:
    # Do NOT use mlflow.start_run() here.
    # ZenML already created the active MLflow run.

    mlflow.log_param(
        "best_model",
        best_model_name
    )

    mlflow.log_metric(
        "best_f1_score",
        best_f1
    )

    mlflow.log_metric(
        "random_forest_f1",
        float(random_forest_f1)
    )

    mlflow.log_metric(
        "svm_f1",
        float(svm_f1)
    )

    mlflow.log_metric(
        "naive_bayes_f1",
        float(naive_bayes_f1)
    )

    # ---------------------------------------------------------
    # 6. Log artifacts
    # ---------------------------------------------------------
    mlflow.log_artifact(
        str(model_path),
        artifact_path="best_model"
    )

    mlflow.log_artifact(
        str(vectorizer_path),
        artifact_path="best_model"
    )

    # ---------------------------------------------------------
    # 7. Print result
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(f"Random Forest F1 : {random_forest_f1:.4f}")
    print(f"Linear SVM F1    : {svm_f1:.4f}")
    print(f"Naive Bayes F1   : {naive_bayes_f1:.4f}")

    print("-" * 60)
    print(f"BEST MODEL       : {best_model_name}")
    print(f"BEST F1 SCORE    : {best_f1:.4f}")
    print(f"MODEL PATH       : {model_path}")
    print("=" * 60 + "\n")

    return str(model_path)



