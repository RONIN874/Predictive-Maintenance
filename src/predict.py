

import os
import sys
import argparse
import numpy as np

# Allow imports from the src directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import load_model


# =============================================================================
# Core prediction function
# =============================================================================

def predict_failure(model, scaler, le, params):
    """
    Predict machine failure for the given parameters.

    Args:
        model  : Trained sklearn classifier (RandomForestClassifier).
        scaler : Fitted StandardScaler.
        le     : Fitted LabelEncoder for the 'Type' column.
        params : dict with keys:
                     'Type'                    (str)   — H, L, or M
                     'Air temperature [K]'     (float)
                     'Process temperature [K]' (float)
                     'Rotational speed [rpm]'  (int)
                     'Torque [Nm]'             (float)
                     'Tool wear [min]'         (int)

    Returns:
        int: 0 (No Failure / Safe) or 1 (Failure Likely)
    """
    # Encode the 'Type' column
    type_encoded = le.transform([params['Type']])[0]

    # Build feature array in the same column order used during training
    features = np.array([[
        type_encoded,
        params['Air temperature [K]'],
        params['Process temperature [K]'],
        params['Rotational speed [rpm]'],
        params['Torque [Nm]'],
        params['Tool wear [min]']
    ]])

    # Scale using the same scaler from training
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]
    return int(prediction)


# =============================================================================
# Interactive input
# =============================================================================

def get_params_interactive():
    """Prompt the user for machine parameters interactively."""
    print(f"\n{'='*40}")
    print("  Machine Failure Prediction System")
    print(f"{'='*40}\n")
    print("Enter machine parameters:\n")

    machine_type = input("  Type (H / L / M)            : ").strip().upper()
    while machine_type not in ('H', 'L', 'M'):
        print("  [!] Invalid type. Please enter H, L, or M.")
        machine_type = input("  Type (H / L / M)            : ").strip().upper()

    air_temp      = float(input("  Air temperature [K]         : ").strip())
    process_temp  = float(input("  Process temperature [K]     : ").strip())
    rpm           = int(input("  Rotational speed [rpm]      : ").strip())
    torque        = float(input("  Torque [Nm]                 : ").strip())
    tool_wear     = int(input("  Tool wear [min]             : ").strip())

    return {
        'Type':                    machine_type,
        'Air temperature [K]':     air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]':  rpm,
        'Torque [Nm]':             torque,
        'Tool wear [min]':         tool_wear,
    }


# =============================================================================
# Display result
# =============================================================================

def display_result(params, prediction):
    """Pretty-print the input parameters and prediction result."""
    print(f"\n{'='*40}")
    print("  Machine Failure Prediction System")
    print(f"{'='*40}\n")

    print("Input Parameters:")
    print(f"  Type               : {params['Type']}")
    print(f"  Air Temperature    : {params['Air temperature [K]']} K")
    print(f"  Process Temperature: {params['Process temperature [K]']} K")
    print(f"  Rotational Speed   : {params['Rotational speed [rpm]']} rpm")
    print(f"  Torque             : {params['Torque [Nm]']} Nm")
    print(f"  Tool Wear          : {params['Tool wear [min]']} min")

    if prediction == 0:
        print("\n✅ Machine Status: Safe")
        print("   The machine is operating within normal parameters.")
    else:
        print("\n⚠️  Warning: Machine Failure Likely!")
        print("   Immediate inspection recommended.")

    print()


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Predict machine failure using the trained model."
    )
    parser.add_argument('--type',         type=str,   help="Machine type (H/L/M)")
    parser.add_argument('--air-temp',     type=float, help="Air temperature [K]")
    parser.add_argument('--process-temp', type=float, help="Process temperature [K]")
    parser.add_argument('--rpm',          type=int,   help="Rotational speed [rpm]")
    parser.add_argument('--torque',       type=float, help="Torque [Nm]")
    parser.add_argument('--tool-wear',    type=int,   help="Tool wear [min]")
    args = parser.parse_args()

    # Resolve paths relative to project root
    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")

    # Step 1 — Load saved artifacts
    model, scaler, le = load_model(models_dir)

    # Step 2 — Get parameters (CLI arguments or interactive)
    cli_values = [args.type, args.air_temp, args.process_temp,
                  args.rpm, args.torque, args.tool_wear]

    if all(v is not None for v in cli_values):
        params = {
            'Type':                    args.type.upper(),
            'Air temperature [K]':     args.air_temp,
            'Process temperature [K]': args.process_temp,
            'Rotational speed [rpm]':  args.rpm,
            'Torque [Nm]':             args.torque,
            'Tool wear [min]':         args.tool_wear,
        }
    else:
        params = get_params_interactive()

    # Steps 3 + 4 — Predict and display
    prediction = predict_failure(model, scaler, le, params)
    display_result(params, prediction)


if __name__ == "__main__":
    main()
