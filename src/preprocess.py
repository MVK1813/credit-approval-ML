# src/preprocess.py

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_and_split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Basic cleaning and train-test split.

    - Replace '?' with NaN
    - Map Class: '+' -> 1 (approved), '-' -> 0 (rejected)
    - Split into train and test sets with stratification.
    """

    # Replace missing value markers
    df = df.replace("?", np.nan)

    # Map target to 0/1
    if set(df["Class"].unique()) == {"+", "-"}:
        df["Class"] = df["Class"].map({"+": 1, "-": 0})

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"].astype(int)

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test
