import mlflow
import optuna

from typing import Annotated
from zenml import step
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("IMDB_Sentiment_Analysis")

@step
def train_naive_bayes(
    X_train,
    X_test,
    y_train,
    y_test,
) -> tuple[
    Annotated[MultinomialNB, "naive_bayes_model"],
    Annotated[float, "naive_bayes_f1"],
]:

    print("\n" + "=" * 60)
    print("NAIVE BAYES TRAINING")
    print("=" * 60)

    def objective(trial):

        alpha = trial.suggest_float(
            "alpha",
            0.001,
            10.0,
            log=True,
        )

        model = MultinomialNB(alpha=alpha)

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

    model = MultinomialNB(
        alpha=best_params["alpha"],
    )

    model.fit(X_train, y_train)

    mlflow.log_params(best_params)
    mlflow.log_metric("f1_score", best_f1)

    print("Naive Bayes training completed.")

    return model, best_f1



