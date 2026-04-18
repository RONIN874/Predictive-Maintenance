import os
import sys
import argparse

# Ensure the project root is on the path so `src.*` imports work
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import load_data, preprocess
from src.eda import plot_distributions, plot_correlation_heatmap, plot_failure_analysis
from src.model_training import split_data, train_model, evaluate_model
from src.model_utils import save_model, load_model
from src.predict import predict_failure, get_params_interactive, display_result



DATA_PATH   = os.path.join(PROJECT_ROOT, "data", "ai4i2020.csv")
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")

def cmd_train(_args):
    """Full pipeline: load → preprocess → EDA → train → evaluate → save."""
    print(f"\n{'='*60}")
    print("  Predictive Maintenance — Training Pipeline")
    print(f"{'='*60}\n")

    # Part 1 — Load data
    df = load_data(DATA_PATH)

    # Part 3 — EDA plots
    print("\n--- Exploratory Data Analysis ---")
    plot_distributions(df, OUTPUT_DIR)
    plot_correlation_heatmap(df, OUTPUT_DIR)
    plot_failure_analysis(df, OUTPUT_DIR)

    # Part 2 — Preprocess
    X_scaled, y, scaler, feature_columns = preprocess(df)

    # Part 4 — Train & evaluate
    X_train, X_test, y_train, y_test = split_data(X_scaled,y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, feature_columns, OUTPUT_DIR)

    # Part 5 — Save
    save_model(model, scaler, output_dir=MODELS_DIR)

    print(f"\n{'='*60}")
    print("  Training pipeline complete.")
    print(f"  Models saved to : {MODELS_DIR}")
    print(f"  Plots saved to  : {OUTPUT_DIR}")
    print(f"{'='*60}\n")


def cmd_predict(args):
    """Load saved model and predict failure for given parameters."""
    model, scaler= load_model(MODELS_DIR)

    # Check if CLI arguments were provided
    cli_values = [args.air_temp, args.process_temp,
                  args.rpm, args.torque, args.tool_wear]

    if all(v is not None for v in cli_values):
        params = {
            'Air temperature [K]':     args.air_temp,
            'Process temperature [K]': args.process_temp,
            'Rotational speed [rpm]':  args.rpm,
            'Torque [Nm]':             args.torque,
            'Tool wear [min]':         args.tool_wear,
        }
    else:
        params = get_params_interactive()

    prediction = predict_failure(model, scaler, params)
    display_result(params, prediction)


def cmd_eda(_args):
    """Run EDA plots only (no training)."""
    print(f"\n{'='*60}")
    print("  Predictive Maintenance — Exploratory Data Analysis")
    print(f"{'='*60}\n")

    df = load_data(DATA_PATH)

    plot_distributions(df, OUTPUT_DIR)
    plot_correlation_heatmap(df, OUTPUT_DIR)
    plot_failure_analysis(df, OUTPUT_DIR)

    print(f"\n[EDA] All plots saved to: {OUTPUT_DIR}\n")



def build_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="🏭 Industrial Machine Failure Prediction System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- train ---
    subparsers.add_parser("train", help="Train the model (full pipeline)")

    # --- predict ---
    pred_parser = subparsers.add_parser("predict", help="Predict machine failure")
    pred_parser.add_argument('--air-temp',     type=float, help="Air temperature [K]")
    pred_parser.add_argument('--process-temp', type=float, help="Process temperature [K]")
    pred_parser.add_argument('--rpm',          type=int,   help="Rotational speed [rpm]")
    pred_parser.add_argument('--torque',       type=float, help="Torque [Nm]")
    pred_parser.add_argument('--tool-wear',    type=int,   help="Tool wear [min]")

    # --- eda ---
    subparsers.add_parser("eda", help="Run exploratory data analysis only")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "train":   cmd_train,
        "predict": cmd_predict,
        "eda":     cmd_eda,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
