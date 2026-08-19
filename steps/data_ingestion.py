from datasets import load_dataset
import pandas as pd
from typing import Annotated
from zenml import step


@step
def ingest_data() -> tuple[
    Annotated[pd.DataFrame, "train_data"],
    Annotated[pd.DataFrame, "test_data"],
]:

    dataset = load_dataset("stanfordnlp/imdb")

    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    return train_df, test_df


