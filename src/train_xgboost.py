"""
train_xgboost.py

Phase 2:
- Use the SAME preprocessing pipeline (same TF-IDF features)
- Train an XGBoost (XGBClassifier) model using ensemble learning
- Evaluate using the same metrics as Random Forest
- Save the trained model as models/xgboost_model.pkl

Run from project root:
    python src/train_xgboost.py
"""

import os
import pickle

from xgboost import XGBClassifier

import config
import data_preprocessing
import utils


def train_xgboost():
    """
    Train and evaluate an XGBoost classifier on the phishing email dataset.
    """
    print("\n[PHASE 2] Training XGBoost classifier...")

    # Reuse the same preprocessing pipeline
    X_train, X_test, y_train, y_test, _, _ = data_preprocessing.preprocess_data()

    # ----------------------------------------------------------------
    # Define the XGBoost model
    # ----------------------------------------------------------------
    # n_estimators   -> number of trees (boosting rounds)
    # max_depth      -> depth of each tree; higher = more complex model
    # learning_rate  -> step size shrinkage; smaller = slower, more precise learning
    # subsample      -> fraction of training instances used in each tree (to reduce overfitting)
    # colsample_bytree -> fraction of features used in each tree
    xgb_clf = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",  # binary classification
        eval_metric="logloss",        # common evaluation metric for binary problems
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )

    print("[INFO] Fitting XGBoost model...")
    xgb_clf.fit(X_train, y_train)

    print("[INFO] Predicting on test set...")
    y_pred = xgb_clf.predict(X_test)

    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------
    metrics = utils.print_classification_metrics(y_test, y_pred, model_name="XGBoost")

    # Optionally plot confusion matrix and save as image
    cm_path = os.path.join(config.MODELS_DIR, "xgboost_confusion_matrix.png")
    utils.plot_confusion_matrix(y_test, y_pred, model_name="XGBoost", save_path=cm_path, show_plot=False)

    # ----------------------------------------------------------------
    # Save the trained model
    # ----------------------------------------------------------------
    model_path = os.path.join(config.MODELS_DIR, "xgboost_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(xgb_clf, f)

    print(f"[INFO] XGBoost model saved to: {model_path}")

    # Print a simple summary of main metrics
    print("\n[SUMMARY] XGBoost performance:")
    print(f"    XGBoost Accuracy : {metrics['accuracy']:.4f}")
    print(f"    XGBoost Precision: {metrics['precision']:.4f}")
    print(f"    XGBoost Recall   : {metrics['recall']:.4f}")
    print(f"    XGBoost F1-score : {metrics['f1']:.4f}")

    return xgb_clf, metrics


if __name__ == "__main__":
    train_xgboost()
