import mlflow
import joblib
from pathlib import Path
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
    # Select best model
    # ---------------------------------------------------------

    scores = {
        "Random Forest": random_forest_f1,
        "Linear SVM": svm_f1,
        "Naive Bayes": naive_bayes_f1,
    }

    models = {
        "Random Forest": random_forest_model,
        "Linear SVM": svm_model,
        "Naive Bayes": naive_bayes_model,
    }

    best_model_name = max(scores, key=scores.get)
    best_model = models[best_model_name]
    best_f1 = scores[best_model_name]

    print("=" * 60)
    print(f"Best Model: {best_model_name}")
    print(f"Best F1 Score: {best_f1:.4f}")
    print("=" * 60)

    # ---------------------------------------------------------
    # Save model + vectorizer
    # ---------------------------------------------------------

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "best_model.joblib"

    model_data = {
        "model": best_model,
        "vectorizer": vectorizer,
        "model_name": best_model_name,
    }

    joblib.dump(model_data, model_path)

    print(f"Model saved to: {model_path}")

    # ---------------------------------------------------------
    # MLflow
    # ---------------------------------------------------------

    if mlflow.active_run():
        mlflow.log_metric("best_f1_score", float(best_f1))
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_artifact(str(model_path))

    return str(model_path)
