"""
evaluate_models.py

Compares the performance of the saved Random Forest and XGBoost models
on the same test set.

This script:
- Re-runs the preprocessing pipeline (same random_state, same TF-IDF settings)
- Loads saved models from models/
- Evaluates both models on the same X_test, y_test
- Prints metrics side-by-side for easy comparison

Run from project root:
    python src/evaluate_models.py
"""

import os
import pickle

import config
import data_preprocessing
import utils


def evaluate_models():
    # ----------------------------------------------------------------
    # 1. Preprocess data to get test set
    # ----------------------------------------------------------------
    print("[INFO] Preprocessing data for evaluation...")
    X_train, X_test, y_train, y_test, _, _ = data_preprocessing.preprocess_data()

    # Note: Because we use the same random_state and pipeline as in training,
    # the TF-IDF feature space and train/test split are consistent.

    # ----------------------------------------------------------------
    # 2. Load saved models
    # ----------------------------------------------------------------
    rf_path = os.path.join(config.MODELS_DIR, "random_forest_model.pkl")
    xgb_path = os.path.join(config.MODELS_DIR, "xgboost_model.pkl")

    if not os.path.exists(rf_path):
        raise FileNotFoundError(
            f"Random Forest model file not found at {rf_path}.\n"
            f"Please run: python src/train_random_forest.py"
        )

    if not os.path.exists(xgb_path):
        raise FileNotFoundError(
            f"XGBoost model file not found at {xgb_path}.\n"
            f"Please run: python src/train_xgboost.py"
        )

    print(f"[INFO] Loading Random Forest model from: {rf_path}")
    with open(rf_path, "rb") as f:
        rf_clf = pickle.load(f)

    print(f"[INFO] Loading XGBoost model from: {xgb_path}")
    with open(xgb_path, "rb") as f:
        xgb_clf = pickle.load(f)

    # ----------------------------------------------------------------
    # 3. Evaluate both models on the same test set
    # ----------------------------------------------------------------
    print("\n=== Evaluating Random Forest on test set ===")
    rf_pred = rf_clf.predict(X_test)
    rf_metrics = utils.print_classification_metrics(y_test, rf_pred, model_name="Random Forest")

    print("\n=== Evaluating XGBoost on test set ===")
    xgb_pred = xgb_clf.predict(X_test)
    xgb_metrics = utils.print_classification_metrics(y_test, xgb_pred, model_name="XGBoost")

    # ----------------------------------------------------------------
    # 4. Print a simple side-by-side comparison
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (higher is better)")
    print("=" * 60)

    metrics_names = ["accuracy", "precision", "recall", "f1"]
    for m in metrics_names:
        print(f"{m.capitalize():<10} | Random Forest: {rf_metrics[m]:.4f} | XGBoost: {xgb_metrics[m]:.4f}")

    print("=" * 60)
    print("Note: Both models were evaluated on the same TF-IDF test set.")


if __name__ == "__main__":
    evaluate_models()
