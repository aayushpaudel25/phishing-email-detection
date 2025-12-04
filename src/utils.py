"""
utils.py

Utility functions used across the project:
- Printing standard classification metrics
- Plotting and saving a confusion matrix
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def print_classification_metrics(y_true, y_pred, model_name: str = "Model"):
    """
    Compute and print standard classification metrics.

    Metrics:
        - Accuracy
        - Precision (binary, positive label assumed to be 1)
        - Recall    (binary)
        - F1-score  (binary)
        - Confusion matrix
        - Classification report

    Returns
    -------
    metrics : dict
        Dictionary containing accuracy, precision, recall, f1.
        Useful for later comparison between models.
    """
    print(f"\n{'=' * 60}")
    print(f"{model_name} - Evaluation Metrics")
    print(f"{'=' * 60}")

    # For binary classification, we treat label '1' as the "positive" class.
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str = None, show_plot: bool = True):
    """
    Plot a confusion matrix using seaborn heatmap.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model (for the plot title)
    save_path : str, optional
        If provided, the plot is saved as an image at this path.
    show_plot : bool
        If True, plt.show() is called to display the plot in a window.
        In some environments (e.g. headless servers) you may want to set this to False.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} - Confusion Matrix")

    if save_path is not None:
        # Make sure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[INFO] Confusion matrix saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()
