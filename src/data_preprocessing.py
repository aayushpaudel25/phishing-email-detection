"""
data_preprocessing.py

This module handles:
- Loading the phishing email dataset from CSV
- Basic text cleaning (lowercasing, dropping empty/very short emails)
- Train/test split
- TF-IDF vectorization of email text

Main function:
    preprocess_data()

Returns:
    X_train, X_test, y_train, y_test, vectorizer, label_encoder
"""

import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

import config


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw CSV from data/raw/ using the path in config.RAW_DATA_PATH.

    You only need to change the file name / column names in config.py.
    """
    if not os.path.exists(config.RAW_DATA_PATH):
        raise FileNotFoundError(
            f"CSV file not found at: {config.RAW_DATA_PATH}\n"
            f"Please place your Kaggle CSV in data/raw/ and update DATA_FILE_NAME in config.py"
        )

    print(f"[INFO] Loading data from: {config.RAW_DATA_PATH}")
    df = pd.read_csv(
    config.RAW_DATA_PATH,
    encoding="latin-1",
    usecols=[0, 1]  # keep only v1 (label) and v2 (text)
    )


    # Quick sanity check for required columns
    if config.TEXT_COLUMN not in df.columns:
        raise KeyError(
            f"TEXT_COLUMN '{config.TEXT_COLUMN}' not found in CSV columns.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Update TEXT_COLUMN in config.py."
        )

    if config.LABEL_COLUMN not in df.columns:
        raise KeyError(
            f"LABEL_COLUMN '{config.LABEL_COLUMN}' not found in CSV columns.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Update LABEL_COLUMN in config.py."
        )

    return df


def basic_text_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform simple text cleaning steps:
    - Drop rows with missing text or labels
    - Lowercase the email text
    - Remove very short emails (length < MIN_EMAIL_LENGTH)
    """
    print("[INFO] Performing basic text cleaning...")

    # Drop rows where text or label is missing (NaN)
    df = df.dropna(subset=[config.TEXT_COLUMN, config.LABEL_COLUMN])

    # Convert text to string and lowercase
    df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].astype(str).str.strip()
    df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].str.lower()

    # Remove rows where email text length is too short
    df = df[df[config.TEXT_COLUMN].str.len() >= config.MIN_EMAIL_LENGTH]

    print(f"[INFO] Data after cleaning: {df.shape[0]} rows")

    return df


def preprocess_data(save_artifacts: bool = True):
    """
    Full preprocessing pipeline:
    1. Load raw CSV using config.RAW_DATA_PATH
    2. Clean text (handle NaN, lowercase, drop very short emails)
    3. Encode labels to integers (0/1) using LabelEncoder
    4. Split into train and test sets
    5. Fit TF-IDF vectorizer on training text and transform both train & test

    Parameters
    ----------
    save_artifacts : bool
        If True, save the fitted TfidfVectorizer and LabelEncoder into
        data/processed/ for reuse.

    Returns
    -------
    X_train : sparse matrix
        TF-IDF features for training set
    X_test : sparse matrix
        TF-IDF features for test set
    y_train : 1D numpy array
        Encoded labels for training set
    y_test : 1D numpy array
        Encoded labels for test set
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer (can be reused for new emails later)
    label_encoder : LabelEncoder
        Fitted label encoder mapping original labels -> integers
    """
    # 1. Load data
    df = load_raw_data()

    # 2. Clean data
    df = basic_text_cleaning(df)

    # Separate features (text) and labels
    texts = df[config.TEXT_COLUMN].values
    labels = df[config.LABEL_COLUMN].values

    # 3. Encode labels into integers (e.g., non-phishing=0, phishing=1)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    print("[INFO] Label encoding mapping:")
    for idx, class_name in enumerate(label_encoder.classes_):
        print(f"    '{class_name}' -> {idx}")
    print("    (For binary classification, 1 is treated as the 'positive' class in metrics.)")

    # 4. Train/test split
    # test_size=0.2 means 20% of data is held out for testing.
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y  # keeps class ratio similar in train and test
    )

    print(f"[INFO] Train set size: {len(X_train_text)}")
    print(f"[INFO] Test set size:  {len(X_test_text)}")

    # 5. TF-IDF vectorization
    print("[INFO] Fitting TF-IDF vectorizer on training text...")
    vectorizer = TfidfVectorizer(
        stop_words="english",         # removes common English stop-words
        max_features=config.MAX_TFIDF_FEATURES  # keeps top-N words by frequency
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print(f"[INFO] TF-IDF matrix (train) shape: {X_train.shape}")
    print(f"[INFO] TF-IDF matrix (test)  shape: {X_test.shape}")

    # Optionally save vectorizer & label encoder to disk for later use
    if save_artifacts:
        vec_path = os.path.join(config.PROCESSED_DATA_DIR, "tfidf_vectorizer.pkl")
        le_path = os.path.join(config.PROCESSED_DATA_DIR, "label_encoder.pkl")

        print(f"[INFO] Saving TF-IDF vectorizer to: {vec_path}")
        with open(vec_path, "wb") as f:
            pickle.dump(vectorizer, f)

        print(f"[INFO] Saving label encoder to: {le_path}")
        with open(le_path, "wb") as f:
            pickle.dump(label_encoder, f)

    return X_train, X_test, y_train, y_test, vectorizer, label_encoder


if __name__ == "__main__":
    """
    Running this file directly lets you quickly test your preprocessing.

    Example:
        python src/data_preprocessing.py
    """
    X_train, X_test, y_train, y_test, _, _ = preprocess_data()
    print("[INFO] Preprocessing complete.")
    print(f"[INFO] X_train shape: {X_train.shape}")
    print(f"[INFO] X_test  shape: {X_test.shape}")
