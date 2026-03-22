"""
model_utils.py — Model Persistence Module
==========================================
Part 5 of the Predictive Maintenance pipeline.

Functions:
    save_model(model, scaler, le, output_dir)  -> None
    load_model(model_dir)                      -> (model, scaler, le)
"""

import os
import pickle


# =============================================================================
# Step 1 — Save trained artefacts
# =============================================================================

def save_model(model, scaler, le, output_dir='models'):
    """
    Persist the trained model, scaler, and label encoder to disk.

    Args:
        model      : Trained sklearn classifier (RandomForestClassifier).
        scaler     : Fitted StandardScaler used during preprocessing.
        le         : Fitted LabelEncoder used for the 'Type' column.
        output_dir : Directory to save the .pkl files (default 'models').

    Saves:
        models/random_forest_model.pkl
        models/scaler.pkl
        models/label_encoder.pkl
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, 'random_forest_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save scaler (needed to transform new inputs the same way)
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Save label encoder (needed to encode 'Type' for new inputs)
    le_path = os.path.join(output_dir, 'label_encoder.pkl')
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)

    print(f"\n{'='*60}")
    print("  Model, scaler, and label encoder saved.")
    print(f"    Directory : {os.path.abspath(output_dir)}")
    print(f"    Model     : {model_path}")
    print(f"    Scaler    : {scaler_path}")
    print(f"    Encoder   : {le_path}")
    print(f"{'='*60}\n")


# =============================================================================
# Step 2 — Load trained artefacts
# =============================================================================

def load_model(model_dir='models'):
    """
    Reload the persisted model, scaler, and label encoder from disk.

    Args:
        model_dir : Directory containing the .pkl files (default 'models').

    Returns:
        tuple: (model, scaler, le)

    Raises:
        FileNotFoundError: If any required .pkl file is missing.
    """
    files = {
        'model':   'random_forest_model.pkl',
        'scaler':  'scaler.pkl',
        'encoder': 'label_encoder.pkl',
    }

    # Verify all files exist before loading
    for label, fname in files.items():
        fpath = os.path.join(model_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"[load_model] Missing {label} file: {os.path.abspath(fpath)}"
            )

    with open(os.path.join(model_dir, files['model']), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(model_dir, files['scaler']), 'rb') as f:
        scaler = pickle.load(f)

    with open(os.path.join(model_dir, files['encoder']), 'rb') as f:
        le = pickle.load(f)

    print(f"[load_model] Model loaded successfully from '{os.path.abspath(model_dir)}'")
    return model, scaler, le


# =============================================================================
# Quick test when run directly
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_preprocessing import load_data, preprocess
    from model_training import split_data, train_model

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "Dataset", "ai4i2020.csv")
    models_dir = os.path.join(base_dir, "models")

    # Part 1 + 2
    df = load_data(csv_path)
    X_scaled, y, scaler, le, feature_columns = preprocess(df)

    # Part 4
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    model = train_model(X_train, y_train)

    # Part 5 — Save
    save_model(model, scaler, le, output_dir=models_dir)

    # Part 5 — Load back and verify
    loaded_model, loaded_scaler, loaded_le = load_model(models_dir)

    # Quick sanity check
    y_pred_original = model.predict(X_test)
    y_pred_loaded = loaded_model.predict(X_test)
    match = (y_pred_original == y_pred_loaded).all()
    print(f"\n[Verify] Predictions match after reload: {match}")
