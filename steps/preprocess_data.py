import pandas as pd
import re
from typing import Annotated
from zenml import step


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@step
def preprocess_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[
    Annotated[pd.Series, "X_train"],
    Annotated[pd.Series, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:

    train_df = train_df.dropna().copy()
    test_df = test_df.dropna().copy()

    train_df["text"] = train_df["text"].apply(clean_text)
    test_df["text"] = test_df["text"].apply(clean_text)

    return (
        train_df["text"],
        test_df["text"],
        train_df["label"],
        test_df["label"],
    )


