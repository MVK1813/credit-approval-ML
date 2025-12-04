# src/load_data.py

import os
import pandas as pd


def load_credit_data(data_path: str = "data/crx.data") -> pd.DataFrame:
    """
    Load the Credit Approval (CRX) dataset from the raw .data file.
    The file has 16 columns: 15 attributes + 1 class label.
    """

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find dataset at {data_path}")

    # Load raw data (comma separated, no header)
    df = pd.read_csv(data_path, header=None)

    print(f"Raw dataset shape: {df.shape}")   # e.g. (690, 16)

    # 15 feature columns A1..A15 + 1 target column 'Class'
    df.columns = [f"A{i}" for i in range(1, 16)] + ["Class"]

    return df
