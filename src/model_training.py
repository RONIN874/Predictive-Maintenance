import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)



# Step 1 — Train / Test Split

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split features and target into train/test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"[Split] Train: {X_train.shape}  |  Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test



# Step 2 — Train the model

def train_model(X_train, y_train, n_estimators=100, random_state=42):
   
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight='balanced'  
    )
    model = MultiOutputClassifier(rf)
    print(f"[Train] Training Multioutput RandomForest")
    model.fit(X_train, y_train)
    print("[Train] Training complete.")
    return model



# Steps 3–6 — Evaluate the model

def evaluate_model(model, X_test, y_test, feature_columns=None, output_dir='outputs'):
    """
    Evaluate the MultiOutput trained model.
    """
    os.makedirs(output_dir, exist_ok=True)

    # -- Step 3: Predictions ---------------------------------------------------
    y_pred = model.predict(X_test)

    # -- Step 4: Accuracy ------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"  Accuracy (Exact Match): {accuracy:.4f}")
    print(f"{'='*60}")

    # -- Step 5: Classification report -----------------------------------------
    target_names = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    print("\n-- Classification Report ----------------------------------------")
    print(report)
    
    # (Confusion Matrix plotting removed: standard plot doesn't support MultiOutput)

    # -- Step 6: Feature importance --------------------------------------------
    if feature_columns is not None:
        # Average the feature importances across all 5 estimators
        importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
        sorted_idx = np.argsort(importances)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            [feature_columns[i] for i in sorted_idx],
            importances[sorted_idx],
            color='steelblue',
            edgecolor='black'
        )
        ax.set_xlabel('Average Importance')
        ax.set_title('Feature Importance (Multi-Output RF)', fontsize=13, fontweight='bold')
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
        'y_pred': y_pred
    }



# Run full pipeline when executed directly
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
    save_model(model, scaler, output_dir=models_dir)
