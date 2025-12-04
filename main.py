# main.py

from src.load_data import load_credit_data
from src.preprocess import clean_and_split_data
from src.train_models import train_models
from src.evaluate import evaluate_models


def main():
    # 1. Load raw data
    print("Loading dataset...")
    df = load_credit_data("data/crx.data")
    print(f"Dataset shape: {df.shape}")

    # 2. Clean and split
    print("Cleaning and splitting data...")
    X_train, X_test, y_train, y_test = clean_and_split_data(df)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 3. Train models
    models = train_models(X_train, y_train)

    # 4. Evaluate models
    metrics_df = evaluate_models(models, X_test, y_test, results_dir="results")
    print("\nFinal metrics summary:")
    print(metrics_df)


if __name__ == "__main__":
    main()
