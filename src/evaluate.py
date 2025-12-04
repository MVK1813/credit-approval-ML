# src/evaluate.py

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)


def evaluate_models(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    results_dir: str = "results",
) -> pd.DataFrame:
    """
    Evaluate trained models on the test set.
    - Prints metrics
    - Saves confusion matrix plots
    - Saves ROC curves
    - Returns DataFrame with metrics
    """

    os.makedirs(results_dir, exist_ok=True)

    metrics_records = []

    for name, model in models.items():
        print(f"\n=== {name.upper()} ===")

        y_pred = model.predict(X_test)

        # Some models may not support predict_proba (but we set SVC with probability=True)
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:, 1]
        else:
            # fallback to decision_function if available
            if hasattr(model, "decision_function"):
                y_scores = model.decision_function(X_test)
            else:
                # if absolutely nothing available, use predictions as scores
                y_scores = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_scores)
        except ValueError:
            auc = float("nan")

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"AUC      : {auc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"{name} - Confusion Matrix")
        cm_path = os.path.join(results_dir, f"{name}_confusion_matrix.png")
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()

        # ROC curve
        try:
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            plt.figure()
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{name} - ROC Curve")
            plt.legend(loc="lower right")
            roc_path = os.path.join(results_dir, f"{name}_roc_curve.png")
            plt.savefig(roc_path, bbox_inches="tight")
            plt.close()
        except ValueError:
            pass

        metrics_records.append(
            {
                "model": name,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "auc": auc,
            }
        )

    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv_path = os.path.join(results_dir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nSaved metrics summary to {metrics_csv_path}")

    return metrics_df
