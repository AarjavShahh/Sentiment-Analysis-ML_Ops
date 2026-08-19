import mlflow
import optuna

from typing import Annotated
from zenml import step
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
@step
def train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
) -> tuple[
    Annotated[RandomForestClassifier, "random_forest_model"],
    Annotated[float, "random_forest_f1"],
]:

    print("\n" + "=" * 60)
    print("RANDOM FOREST TRAINING")
    print("=" * 60)

    def objective(trial):

        n_estimators = trial.suggest_int(
            "n_estimators",
            100,
            300,
        )

        max_depth = trial.suggest_int(
            "max_depth",
            5,
            30,
        )

        min_samples_split = trial.suggest_int(
            "min_samples_split",
            2,
            10,
        )

        min_samples_leaf = trial.suggest_int(
            "min_samples_leaf",
            1,
            5,
        )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
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

    model = RandomForestClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    mlflow.log_params(best_params)
    mlflow.log_metric("f1_score", best_f1)

    print("Random Forest training completed.")

    return model, best_f1


