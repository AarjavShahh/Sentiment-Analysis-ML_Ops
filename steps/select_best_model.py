from typing import Annotated

from zenml import step


@step
def select_best_model(
    random_forest_model,
    random_forest_f1: float,
    svm_model,
    svm_f1: float,
    naive_bayes_model,
    naive_bayes_f1: float,
) -> tuple[
    Annotated[object, "best_model"],
    Annotated[str, "best_model_name"],
    Annotated[float, "best_model_f1"],
]:

    scores = {
        "Random Forest": random_forest_f1,
        "SVM": svm_f1,
        "Naive Bayes": naive_bayes_f1,
    }

    print("\n==============================")
    print("MODEL COMPARISON")
    print("==============================")

    for name, score in scores.items():
        print(f"{name}: F1 = {score:.4f}")

    best_model_name = max(
        scores,
        key=scores.get,
    )

    best_model_f1 = scores[best_model_name]

    if best_model_name == "Random Forest":
        best_model = random_forest_model

    elif best_model_name == "SVM":
        best_model = svm_model

    else:
        best_model = naive_bayes_model

    print("\n==============================")
    print("BEST MODEL")
    print("==============================")
    print(f"Model: {best_model_name}")
    print(f"F1 Score: {best_model_f1:.4f}")

    return (
        best_model,
        best_model_name,
        best_model_f1,
    )
