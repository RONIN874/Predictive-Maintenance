import os
import pickle

# =============================================================================
# Step 1 — Save trained artefacts
# =============================================================================

def save_model(model, scaler, output_dir='models'):
    """
    Persist the trained model and scaler to disk.
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

    print(f"\n{'='*60}")
    print("  Model and scaler saved.")
    print(f"    Directory : {os.path.abspath(output_dir)}")
    print(f"    Model     : {model_path}")
    print(f"    Scaler    : {scaler_path}")
    print(f"{'='*60}\n")


# =============================================================================
# Step 2 — Load trained artefacts
# =============================================================================

def load_model(model_dir='models'):
    """
    Reload the persisted model and scaler from disk.
    """
    files = {
        'model':   'random_forest_model.pkl',
        'scaler':  'scaler.pkl',
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

    print(f"[load_model] Model loaded successfully from '{os.path.abspath(model_dir)}'")
    return model, scaler