import mlflow
import optuna

from typing import Annotated
from zenml import step
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("IMDB_Sentiment_Analysis")

@step
def train_svm(
    X_train,
    X_test,
    y_train,
    y_test,
) -> tuple[
    Annotated[LinearSVC, "svm_model"],
    Annotated[float, "svm_f1"],
]:

    print("\n" + "=" * 60)
    print("LINEAR SVM TRAINING")
    print("=" * 60)

    def objective(trial):

        C = trial.suggest_float(
            "C",
            0.01,
            10.0,
            log=True,
        )

        model = LinearSVC(
            C=C,
            random_state=42,
            max_iter=5000,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        return f1_score(
            y_test,
            predictions,
            average="binary",
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        objective,
        n_trials=10,
    )

    best_params = study.best_params
    best_f1 = study.best_value

    print(f"Best parameters: {best_params}")
    print(f"Best F1: {best_f1:.4f}")

    model = LinearSVC(
        C=best_params["C"],
        random_state=42,
        max_iter=5000,
    )

    model.fit(X_train, y_train)

    mlflow.log_params(best_params)
    mlflow.log_metric("f1_score", best_f1)

    print("SVM training completed.")

    return model, best_f1



