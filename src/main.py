"""
main.py

Simple command-line menu to run different parts of the project:

1. Preprocess data only (sanity check)
2. Train Random Forest
3. Train XGBoost
4. Evaluate both models (Random Forest vs XGBoost)

Run from project root:
    python src/main.py
"""

import sys

import data_preprocessing
from train_random_forest import train_random_forest
from train_xgboost import train_xgboost
from evaluate_models import evaluate_models


def print_menu():
    print("\n=== Phishing Email Detection - Main Menu ===")
    print("1. Preprocess data")
    print("2. Train Random Forest")
    print("3. Train XGBoost")
    print("4. Evaluate both models (Random Forest vs XGBoost)")
    print("0. Exit")


def main():
    while True:
        print_menu()
        choice = input("Enter your choice (0-4): ").strip()

        if choice == "1":
            print("\n[OPTION 1] Preprocessing data...")
            X_train, X_test, y_train, y_test, _, _ = data_preprocessing.preprocess_data()
            print("[INFO] Preprocessing finished.")
            print(f"[INFO] X_train shape: {X_train.shape}")
            print(f"[INFO] X_test  shape: {X_test.shape}")
            print(f"[INFO] y_train size: {len(y_train)}")
            print(f"[INFO] y_test  size: {len(y_test)}")

        elif choice == "2":
            print("\n[OPTION 2] Training Random Forest...")
            train_random_forest()

        elif choice == "3":
            print("\n[OPTION 3] Training XGBoost...")
            train_xgboost()

        elif choice == "4":
            print("\n[OPTION 4] Evaluating saved models...")
            evaluate_models()

        elif choice == "0":
            print("Exiting...")
            sys.exit(0)

        else:
            print("Invalid choice. Please enter a number between 0 and 4.")


if __name__ == "__main__":
    main()
