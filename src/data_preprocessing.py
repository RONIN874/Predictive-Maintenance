"""
data_preprocessing.py — Dataset Handling & Preprocessing Module
================================================================
Part 1 + Part 2 of the Predictive Maintenance pipeline.

Functions:
    load_data(path)     -> pd.DataFrame
    preprocess(df)      -> (X_scaled, y, scaler, feature_columns)
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


# =============================================================================
# PART 1 — Load & Inspect
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"Dataset not found at: {abs_path}\n"
            "Please check the path and try again."
        )

    print(f"Loading dataset from: {abs_path}")
    df = pd.read_csv(abs_path)
    print(f"\n{'='*60}")
    print(f"  Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"{'='*60}\n")

    # -- Step 2: Inspect the dataset ------------------------------------------
    print("-- First 5 rows -------------------------------------------------")
    print(df.head().to_string())

    print("\n-- Column names and data types ----------------------------------")
    print(df.info())

    print("\n-- Missing values per column ------------------------------------")
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print("  [OK] No missing values found.")
    else:
        print(null_counts[null_counts > 0])

    # -- Step 3: Basic statistics ---------------------------------------------
    print("\n-- Descriptive statistics (numeric columns) ---------------------")
    print(df.describe().to_string())

    print("\n-- Target variable distribution: 'Machine failure' --------------")
    counts = df['Machine failure'].value_counts()
    total = len(df)
    for label, count in counts.items():
        tag = "Failure" if label == 1 else "No Failure"
        print(f"  {label} ({tag:>10}): {count:>6,}  ({count / total * 100:.2f}%)")

    print(f"\n{'='*60}")
    print("  Dataset loaded successfully.")
    print(f"{'='*60}\n")

    return df



# PART 2 — Preprocess

def preprocess(df: pd.DataFrame):
   
    df = df.copy()  # do not mutate the original

    # -- Step 1: Drop identifier columns 
    columns_to_drop = ['UDI', 'Product ID' , 'Type']
    df.drop(columns=columns_to_drop, inplace=True)
    print(f"[Step 1] Dropped columns: {columns_to_drop}")

    # -- Step 2: Drop failure-type flags (target leakage) 
    target_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    y = df[target_columns]
    # -- Step 3: Handle missing values 
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    dropped = before - after
    if dropped:
        print(f"[Step 3] Dropped {dropped} rows with null values.")
    else:
        print(f"[Step 3] No null rows found. Rows retained: {after:,}")

   
   

    # -- Step 4: Feature selection 
    feature_columns = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    X = df[feature_columns]
    print(f"[Step 4] Features selected: {feature_columns}")
    print(f"         X shape: {X.shape}  |  y shape: {y.shape}")

    # -- Step : Feature scaling 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
   
    print("  Preprocessing complete.")

    return X_scaled, y, scaler, feature_columns


# Quick test when run directly
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "Dataset", "ai4i2020.csv")

    dataframe = load_data(csv_path)
    X_scaled, y, scaler, features = preprocess(dataframe)

    print(f"Returned X_scaled shape : {X_scaled.shape}")
    print(f"Returned y shape        : {y.shape}")
    print(f"Scaler type             : {type(scaler).__name__}")
    print(f"Feature columns         : {features}")
