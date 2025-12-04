"""
config.py

All project-wide configuration lives here, so you only need to
edit paths / column names / basic parameters in ONE place.
"""

import os

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

# BASE_DIR will be the root folder "phishing_email_detection"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Models directory
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Make sure these folders exist (safe even if already present)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Dataset configuration
# -------------------------------------------------------------------


DATA_FILE_NAME = "spam.csv"

RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, DATA_FILE_NAME)


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
TEXT_COLUMN = "v2"   # column with full email text
LABEL_COLUMN = "v1"       # column with phishing / non-phishing label

# -------------------------------------------------------------------
# Train/test split & randomness
# -------------------------------------------------------------------

# Test set size = 20% of all data (you can change but 0.2 is standard)
TEST_SIZE = 0.2

# Random seed to make experiments reproducible
RANDOM_STATE = 42

# Minimum number of characters to keep an email.
# Very short emails may be noise or not useful.
MIN_EMAIL_LENGTH = 20

# Maximum number of features for TF-IDF.
# Higher = more vocabulary, more memory; 5000 is reasonable.
MAX_TFIDF_FEATURES = 5000
