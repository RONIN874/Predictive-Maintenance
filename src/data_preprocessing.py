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
    """
    Load, inspect, and return the AI4I 2020 dataset as a DataFrame.

    Steps performed:
        1. Load CSV from the given path
        2. Print shape, column info, and null counts
        3. Print descriptive statistics for numeric columns
        4. Print target variable (Machine failure) distribution

    Args:
        path (str): Relative or absolute path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the CSV file does not exist at the given path.
    """
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


# =============================================================================
# PART 2 — Preprocess
# =============================================================================

def preprocess(df: pd.DataFrame):
    """
    Clean and prepare the raw DataFrame for model training.

    Steps performed:
        1. Drop non-predictive identifier columns (UDI, Product ID)
        2. Drop individual failure-type flags to prevent target leakage
           (TWF, HDF, PWF, OSF, RNF)
        3. Handle missing values (drop rows; dataset has 0 nulls but defensive)
        4. Encode categorical column 'Type' with LabelEncoder (H=0, L=1, M=2)
        5. Select feature columns and define target (y)
        6. Scale features with StandardScaler

    Args:
        df (pd.DataFrame): Raw DataFrame returned by load_data().

    Returns:
        tuple:
            X_scaled        (np.ndarray)       : Scaled feature matrix
            y               (pd.Series)        : Binary target (Machine failure)
            scaler          (StandardScaler)   : Fitted scaler (reuse for inference)
            le              (LabelEncoder)     : Fitted encoder for 'Type' column
            feature_columns (list[str])        : Ordered list of feature names
    """
    df = df.copy()  # do not mutate the original

    # -- Step 1: Drop identifier columns --------------------------------------
    columns_to_drop = ['UDI', 'Product ID']
    df.drop(columns=columns_to_drop, inplace=True)
    print(f"[Step 1] Dropped identifier columns: {columns_to_drop}")

    # -- Step 2: Drop failure-type flags (target leakage) ----------------------
    leakage_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df.drop(columns=leakage_cols, inplace=True)
    print(f"[Step 2] Dropped leakage columns:    {leakage_cols}")
    print(f"         Remaining columns: {list(df.columns)}")

    # -- Step 3: Handle missing values ----------------------------------------
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    dropped = before - after
    if dropped:
        print(f"[Step 3] Dropped {dropped} rows with null values.")
    else:
        print(f"[Step 3] No null rows found. Rows retained: {after:,}")

    # -- Step 4: Encode categorical variable 'Type' ---------------------------
    le = LabelEncoder()
    df['Type'] = le.fit_transform(df['Type'])
    print(f"[Step 4] LabelEncoder applied to 'Type'.")
    print(f"         Classes: {list(le.classes_)}  ->  {list(le.transform(le.classes_))}")

    # -- Step 5: Feature selection --------------------------------------------
    feature_columns = [
        'Type',
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    X = df[feature_columns]
    y = df['Machine failure']
    print(f"[Step 5] Features selected: {feature_columns}")
    print(f"         X shape: {X.shape}  |  y shape: {y.shape}")
    print(f"         Target distribution:\n{y.value_counts().to_string()}")

    # -- Step 6: Feature scaling ----------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"[Step 6] StandardScaler applied.")
    print(f"         Scaled X mean (should be ~0): {X_scaled.mean(axis=0).round(4)}")
    print(f"         Scaled X std  (should be ~1): {X_scaled.std(axis=0).round(4)}")

    print(f"\n{'='*60}")
    print("  Preprocessing complete.")
    print(f"{'='*60}\n")

    return X_scaled, y, scaler, le, feature_columns


# =============================================================================
# Quick test when run directly
# =============================================================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "Dataset", "ai4i2020.csv")

    dataframe = load_data(csv_path)
    X_scaled, y, scaler, le, features = preprocess(dataframe)

    print(f"Returned X_scaled shape : {X_scaled.shape}")
    print(f"Returned y shape        : {y.shape}")
    print(f"Scaler type             : {type(scaler).__name__}")
    print(f"LabelEncoder classes    : {list(le.classes_)}")
    print(f"Feature columns         : {features}")
