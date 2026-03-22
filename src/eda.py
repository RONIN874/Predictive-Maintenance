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
    """
    Plot a Pearson correlation heatmap for numeric features + target.

    Insight: Reveals multicollinearity (e.g. Torque vs Rotational Speed are
             inversely correlated) and feature-target relationships.

    Args:
        df         (pd.DataFrame): Raw or minimally processed dataset.
        output_dir (str)         : Directory to save the plot.

    Returns:
        str: Absolute path to the saved PNG file.
    """
    _ensure_output_dir(output_dir)

    corr_cols = NUMERIC_COLS + ['Machine failure']
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        linewidths=0.5,
        ax=ax,
        square=True
    )
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=12)
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
    """
    Plot class distribution and feature box plots grouped by failure status.

    Insight: Highlights which features differ most between failure and
             non-failure groups (e.g. higher torque -> more failures).

    Args:
        df         (pd.DataFrame): Raw or minimally processed dataset.
        output_dir (str)         : Directory to save the plots.

    Returns:
        tuple[str, str]: Absolute paths to failure_distribution.png and
                         failure_boxplots.png respectively.
    """
    _ensure_output_dir(output_dir)

    # --- Count plot: failure vs non-failure -----------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    palette = {'0': '#4caf50', '1': '#f44336'}  # Green = OK, Red = Failure
    plot_df = df.copy()
    plot_df['Machine failure'] = plot_df['Machine failure'].astype(str)
    sns.countplot(
        x='Machine failure',
        data=plot_df,
        hue='Machine failure',
        palette=palette,
        ax=ax,
        edgecolor='black',
        legend=False
    )
    ax.set_title('Machine Failure Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Machine Failure (0 = No Failure, 1 = Failure)')
    ax.set_ylabel('Count')

    # Annotate bars with counts and percentages
    total = len(plot_df)
    for p in ax.patches:
        count = int(p.get_height())
        ax.annotate(
            f'{count:,}\n({count/total*100:.1f}%)',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    plt.tight_layout()
    dist_path = os.path.join(output_dir, 'failure_distribution.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[EDA] Saved: {os.path.abspath(dist_path)}")

    # --- Box plots: feature values grouped by failure ------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Feature Distribution by Failure Status', fontsize=14, fontweight='bold')

    for i, col in enumerate(NUMERIC_COLS):
        ax = axes[i // 3][i % 3]
        sns.boxplot(
            x='Machine failure',
            y=col,
            data=plot_df,
            hue='Machine failure',
            palette=palette,
            ax=ax,
            legend=False
        )
        ax.set_title(col, fontsize=10)
        ax.set_xlabel('Machine Failure')

    # Hide 6th unused subplot
    axes[1][2].set_visible(False)

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
