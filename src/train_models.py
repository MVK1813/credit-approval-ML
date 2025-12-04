# src/train_models.py

from typing import Dict
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def build_preprocessor(feature_names):
    numeric_features = ["A2", "A3", "A8", "A11", "A14"]
    categorical_features = [f for f in feature_names if f not in numeric_features]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor


def get_model_pipelines(feature_names) -> Dict[str, Pipeline]:
    preprocessor = build_preprocessor(feature_names)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "svm_rbf": SVC(kernel="rbf", probability=True, class_weight="balanced"),
        "knn": KNeighborsClassifier(n_neighbors=5)
    }

    pipelines = {}
    for name, model in models.items():
        pipelines[name] = Pipeline([("preprocess", preprocessor), ("model", model)])

    return pipelines


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Pipeline]:
    feature_names = list(X_train.columns)
    pipelines = get_model_pipelines(feature_names)

    print("\nTraining Models...\n")
    for name, pipe in pipelines.items():
        print(f"> Training {name} ...")
        pipe.fit(X_train, y_train)

    print("\n✔ All models trained successfully.")
    return pipelines
