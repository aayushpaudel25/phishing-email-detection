"""
train_random_forest.py

Phase 1:
- Use the preprocessing pipeline to get TF-IDF features
- Train a RandomForestClassifier for phishing email detection
- Evaluate using standard metrics
- Optionally plot and save the confusion matrix
- Save the trained Random Forest model as models/random_forest_model.pkl

Run from project root:
    python src/train_random_forest.py
"""

import os
import pickle

from sklearn.ensemble import RandomForestClassifier

import config
import data_preprocessing
import utils


def train_random_forest():
    """
    Train and evaluate a Random Forest classifier on the phishing email dataset.
    """
    print("\n[PHASE 1] Training Random Forest classifier...")

    # Get preprocessed data
    X_train, X_test, y_train, y_test, _, _ = data_preprocessing.preprocess_data()

    # ----------------------------------------------------------------
    # Define the Random Forest model
    # ----------------------------------------------------------------
    # n_estimators=100 -> number of trees in the forest
    # max_depth=None  -> trees grow until all leaves are pure or contain < min_samples_split samples
    # n_jobs=-1       -> use all CPU cores for speed
    # random_state    -> for reproducibility
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )

    print("[INFO] Fitting Random Forest model...")
    rf_clf.fit(X_train, y_train)

    print("[INFO] Predicting on test set...")
    y_pred = rf_clf.predict(X_test)

    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------
    metrics = utils.print_classification_metrics(y_test, y_pred, model_name="Random Forest")

    # Optionally plot confusion matrix and save as image
    cm_path = os.path.join(config.MODELS_DIR, "random_forest_confusion_matrix.png")
    utils.plot_confusion_matrix(y_test, y_pred, model_name="Random Forest", save_path=cm_path, show_plot=False)

    # ----------------------------------------------------------------
    # Save the trained model
    # ----------------------------------------------------------------
    model_path = os.path.join(config.MODELS_DIR, "random_forest_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(rf_clf, f)

    print(f"[INFO] Random Forest model saved to: {model_path}")

    # Print a simple summary of main metrics
    print("\n[SUMMARY] Random Forest performance:")
    print(f"    Accuracy : {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall   : {metrics['recall']:.4f}")
    print(f"    F1-score : {metrics['f1']:.4f}")

    return rf_clf, metrics


if __name__ == "__main__":
    train_random_forest()
