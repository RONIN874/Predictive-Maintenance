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

def predict_failure(model, scaler, params):
    
    # Build feature array in the same column order used during training
    features = np.array([[
        params['Air temperature [K]'],
        params['Process temperature [K]'],
        params['Rotational speed [rpm]'],
        params['Torque [Nm]'],
        params['Tool wear [min]']
    ]])

    # Scale using the same scaler from training
    features_scaled = scaler.transform(features)

    # Predict class and probability
    prediction = (model.predict(features_scaled)[0])
    
    # Get top contributing factors if supported
    
    failure_names = ['Tool Wear Failure', 'Heat Dissipation Failure', 
                     'Power Failure', 'Overstrain Failure', 'Random Failures']
    active_failure = [failure_names[i] for i, p in enumerate(prediction) if p == 1]
    return active_failure



# Interactive input

def get_params_interactive():
    """Prompt the user for machine parameters interactively."""
    print(f"\n{'='*40}")
    print("  Machine Failure Prediction System")
    print(f"{'='*40}\n")


    air_temp      = float(input("  Air temperature [K]         : ").strip())
    process_temp  = float(input("  Process temperature [K]     : ").strip())
    rpm           = int(input("  Rotational speed [rpm]      : ").strip())
    torque        = float(input("  Torque [Nm]                 : ").strip())
    tool_wear     = int(input("  Tool wear [min]             : ").strip())

    return {
        'Air temperature [K]':     air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]':  rpm,
        'Torque [Nm]':             torque,
        'Tool wear [min]':         tool_wear,
    }


# Display result

def display_result(params, active_failures):
    print(f"\n{'='*40}")
    print("  Multi-Output Failure Prediction")
    print(f"{'='*40}\n")
    
    # This is what I meant by "print your 5 numeric params here"
    print("Input Parameters:")
    print(f"  Air Temperature    : {params['Air temperature [K]']} K")
    print(f"  Process Temperature: {params['Process temperature [K]']} K")
    print(f"  Rotational Speed   : {params['Rotational speed [rpm]']} rpm")
    print(f"  Torque             : {params['Torque [Nm]']} Nm")
    print(f"  Tool Wear          : {params['Tool wear [min]']} min")
    
    if not active_failures:
        print("\n✅ Machine Status: Safe (No failures predicted)")
    else:
        print("\n⚠️  WARNING: Potential Failures Detected!")
        for f in active_failures:
            print(f"  -> {f}")
    print()

# CLI entry point

def main():
    parser = argparse.ArgumentParser(
        description="Predict machine failure using the trained model."
    )
    # Removed the --type argument completely
    parser.add_argument('--air-temp',     type=float, help="Air temperature [K]")
    parser.add_argument('--process-temp', type=float, help="Process temperature [K]")
    parser.add_argument('--rpm',          type=int,   help="Rotational speed [rpm]")
    parser.add_argument('--torque',       type=float, help="Torque [Nm]")
    parser.add_argument('--tool-wear',    type=int,   help="Tool wear [min]")
    args = parser.parse_args()

    # Resolve paths relative to project root
    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")

    # Step 1 — Load saved artifacts (Removed 'le')
    model, scaler = load_model(models_dir)

    # Step 2 — Get parameters (Removed args.type)
    cli_values = [args.air_temp, args.process_temp,
                  args.rpm, args.torque, args.tool_wear]

    if all(v is not None for v in cli_values):
        # Removed 'Type' from this dict
        params = {
            'Air temperature [K]':     args.air_temp,
            'Process temperature [K]': args.process_temp,
            'Rotational speed [rpm]':  args.rpm,
            'Torque [Nm]':             args.torque,
            'Tool wear [min]':         args.tool_wear,
        }
    else:
        params = get_params_interactive()

    # Steps 3 + 4 — Predict and display
    # Matched to your updated predict_failure function
    active_failures = predict_failure(model, scaler, params)
    display_result(params, active_failures)


if __name__ == "__main__":
    main()
