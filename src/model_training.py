"""
model_training.py — ML Model Training & Evaluation Module
==========================================================
Part 4 of the Predictive Maintenance pipeline.

Functions:
    split_data(X, y)                              -> (X_train, X_test, y_train, y_test)
    train_model(X_train, y_train)                 -> RandomForestClassifier
    evaluate_model(model, X_test, y_test, ...)    -> dict with metrics
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =============================================================================
# Step 1 — Train / Test Split
# =============================================================================

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split features and target into train/test sets (stratified).

    Args:
        X             : Feature matrix (np.ndarray or pd.DataFrame).
        y             : Target vector (pd.Series or np.ndarray).
        test_size     : Fraction of data reserved for testing (default 0.2).
        random_state  : Seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[Split] Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"[Split] Train failure rate: {y_train.mean():.4f}  |  Test failure rate: {y_test.mean():.4f}")
    return X_train, X_test, y_train, y_test


# =============================================================================
# Step 2 — Train the model
# =============================================================================

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier with balanced class weights.

    Args:
        X_train       : Training feature matrix.
        y_train       : Training target vector.
        n_estimators  : Number of trees (default 100).
        random_state  : Seed for reproducibility.

    Returns:
        RandomForestClassifier: Fitted model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight='balanced'  # handles class imbalance
    )
    print(f"[Train] Training RandomForest (n_estimators={n_estimators}, class_weight='balanced')...")
    model.fit(X_train, y_train)
    print("[Train] Training complete.")
    return model


# =============================================================================
# Steps 3–6 — Evaluate the model
# =============================================================================

def evaluate_model(model, X_test, y_test, feature_columns=None, output_dir='outputs'):
    """
    Evaluate the trained model: accuracy, classification report,
    confusion matrix plot, and feature importance plot.

    Args:
        model            : Trained sklearn classifier.
        X_test           : Test feature matrix.
        y_test           : Test target vector.
        feature_columns  : List of feature names (for importance plot).
        output_dir       : Directory to save plots.

    Returns:
        dict: {
            'accuracy'   : float,
            'report'     : str   (classification report text),
            'y_pred'     : np.ndarray,
            'confusion'  : np.ndarray
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    # -- Step 3: Predictions ---------------------------------------------------
    y_pred = model.predict(X_test)

    # -- Step 4: Accuracy ------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"{'='*60}")

    # -- Step 5: Classification report & confusion matrix ----------------------
    report = classification_report(y_test, y_pred, target_names=['No Failure', 'Failure'])
    print("\n-- Classification Report ----------------------------------------")
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    print("-- Confusion Matrix (raw) ---------------------------------------")
    print(cm)

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Failure', 'Failure'])
    disp.plot(cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Eval] Saved: {os.path.abspath(cm_path)}")

    # -- Step 6: Feature importance --------------------------------------------
    if feature_columns is not None:
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            [feature_columns[i] for i in sorted_idx],
            importances[sorted_idx],
            color='steelblue',
            edgecolor='black'
        )
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance (Random Forest)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        fi_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(fi_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Eval] Saved: {os.path.abspath(fi_path)}")

        # Print ranked features
        print("\n-- Feature Importance (ranked) ----------------------------------")
        for i in reversed(sorted_idx):
            print(f"  {feature_columns[i]:30s} : {importances[i]:.4f}")

    print(f"\n{'='*60}")
    print("  Evaluation complete.")
    print(f"{'='*60}\n")

    return {
        'accuracy': accuracy,
        'report': report,
        'y_pred': y_pred,
        'confusion': cm
    }


# =============================================================================
# Run full pipeline when executed directly
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_preprocessing import load_data, preprocess
    from model_utils import save_model

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "Dataset", "ai4i2020.csv")
    output_dir = os.path.join(base_dir, "outputs")
    models_dir = os.path.join(base_dir, "models")

    # Part 1 + 2
    df = load_data(csv_path)
    X_scaled, y, scaler, le, feature_columns = preprocess(df)

    # Part 4
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    model = train_model(X_train, y_train)
    results = evaluate_model(model, X_test, y_test, feature_columns, output_dir)

    # Part 5 — Save model
    save_model(model, scaler, le, output_dir=models_dir)
