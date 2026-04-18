"""
eda.py — Exploratory Data Analysis Module
==========================================
Part 3 of the Predictive Maintenance pipeline.

Functions:
    plot_distributions(df, output_dir)       -> saves feature_distributions.png
    plot_correlation_heatmap(df, output_dir) -> saves correlation_heatmap.png
    plot_failure_analysis(df, output_dir)    -> saves failure_distribution.png
                                                       failure_boxplots.png
"""

import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe for script execution
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Consistent style across all plots
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 100

# Numeric feature columns used in every plot
NUMERIC_COLS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]


def _ensure_output_dir(output_dir: str) -> str:
    """Create the output directory if it does not already exist."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# =============================================================================
# Step 1 — Feature distribution plots
# =============================================================================

def plot_distributions(df: pd.DataFrame, output_dir: str = 'outputs') -> str:
    """
    Plot histogram subplots for each numeric feature.

    Insight: Shows if features are normally distributed, skewed, or have outliers.

    Args:
        df         (pd.DataFrame): Raw or minimally processed dataset.
        output_dir (str)         : Directory to save the plot.

    Returns:
        str: Absolute path to the saved PNG file.
    """
    _ensure_output_dir(output_dir)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Feature Distributions — AI4I 2020 Dataset', fontsize=14, fontweight='bold')

    for i, col in enumerate(NUMERIC_COLS):
        ax = axes[i // 3][i % 3]
        ax.hist(df[col], bins=30, edgecolor='black', alpha=0.7, color=sns.color_palette('muted')[i])
        ax.set_title(col, fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')

    # Hide the unused 6th subplot (2×3 grid, 5 features)
    axes[1][2].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'feature_distributions.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[EDA] Saved: {os.path.abspath(out_path)}")
    return os.path.abspath(out_path)


# =============================================================================
# Step 2 — Correlation heatmap
# =============================================================================

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str = 'outputs') -> str:
    _ensure_output_dir(output_dir)

    # Use the 5 new targets instead of 'Machine failure'
    target_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    corr_cols = NUMERIC_COLS + target_cols
    
    # We need to make sure these columns exist in the raw dataframe for EDA
    existing_cols = [c for c in corr_cols if c in df.columns]
    corr = df[existing_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10)) # Made slightly larger to fit new columns
    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        linewidths=0.5,
        ax=ax,
        square=True
    )
    ax.set_title('Feature Correlation Heatmap (Multi-Output)', fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[EDA] Saved: {os.path.abspath(out_path)}")
    return os.path.abspath(out_path)


# =============================================================================
# Step 3 — Failure analysis
# =============================================================================

def plot_failure_analysis(df: pd.DataFrame, output_dir: str = 'outputs') -> tuple:
    _ensure_output_dir(output_dir)
    target_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    plot_df = df.copy()

    # --- Count plot: Frequency of EACH failure type --------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Sum up the occurrences of each failure type
    failure_counts = plot_df[target_cols].sum().sort_values(ascending=False)
    
    sns.barplot(x=failure_counts.index, y=failure_counts.values, ax=ax, palette='Reds_r', edgecolor='black')
    ax.set_title('Distribution of Specific Failure Types', fontsize=13, fontweight='bold')
    ax.set_xlabel('Failure Type')
    ax.set_ylabel('Total Occurrences')

    # Annotate bars
    for p in ax.patches:
        count = int(p.get_height())
        ax.annotate(f'{count:,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    dist_path = os.path.join(output_dir, 'failure_distribution.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[EDA] Saved: {os.path.abspath(dist_path)}")

    # --- Box plots: grouped by "Any Failure" ---------------------------------
    # Create a temporary column that is 1 if ANY failure occurred, 0 otherwise
    plot_df['Any Failure'] = plot_df[target_cols].max(axis=1).astype(str)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Feature Distribution by Failure Status (Any Failure)', fontsize=14, fontweight='bold')
    palette = {'0': '#4caf50', '1': '#f44336'}

    for i, col in enumerate(NUMERIC_COLS):
        ax = axes[i // 3][i % 3]
        sns.boxplot(
            x='Any Failure', y=col, data=plot_df, hue='Any Failure', 
            palette=palette, ax=ax, legend=False
        )
        ax.set_title(col, fontsize=10)
        ax.set_xlabel('Any Failure (0 = Safe, 1 = Failed)')

    axes[1][2].set_visible(False) # Hide 6th unused subplot

    plt.tight_layout()
    box_path = os.path.join(output_dir, 'failure_boxplots.png')
    plt.savefig(box_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[EDA] Saved: {os.path.abspath(box_path)}")

    return os.path.abspath(dist_path), os.path.abspath(box_path)


# =============================================================================
# Run all EDA when executed directly
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_preprocessing import load_data

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "Dataset", "ai4i2020.csv")
    output_dir = os.path.join(base_dir, "outputs")

    df = load_data(csv_path)

    print("\nRunning EDA...")
    plot_distributions(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_failure_analysis(df, output_dir)
    print("\n[EDA] All plots saved to:", output_dir)
