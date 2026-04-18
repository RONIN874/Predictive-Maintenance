import os
import pickle
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# ─── Load artefacts once at startup ──────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

FAILURE_LABELS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FAILURE_FULL   = [
    "Tool Wear Failure",
    "Heat Dissipation Failure",
    "Power Failure",
    "Overstrain Failure",
    "Random Failure",
]


# ─── Prediction logic ────────────────────────────────────────────────────────

def predict(air_temp, process_temp, rpm, torque, tool_wear):
    """Return status markdown, probability table (DataFrame), and bar chart."""

    features = np.array([[air_temp, process_temp, rpm, torque, tool_wear]])
    features_scaled = scaler.transform(features)

    # Binary predictions (shape: 1 × 5)
    preds = model.predict(features_scaled)[0]

    # Per-target probability of class 1
    probs = []
    for estimator in model.estimators_:
        p = estimator.predict_proba(features_scaled)[0]
        # Some estimators may only have seen one class
        if len(estimator.classes_) == 1:
            prob_1 = 0.0 if estimator.classes_[0] == 0 else 1.0
        else:
            prob_1 = float(p[1])
        probs.append(prob_1)

    max_prob = max(probs)

    # ── Status markdown ──────────────────────────────────────────────────
    if max_prob > 0.70:
        status_badge  = "🔴  HIGH RISK"
        badge_color   = "#d32f2f"
        card_bg       = "#ffe5e5"
        card_border   = "#d32f2f"
        action_text   = "⚡ Immediate maintenance required"
    elif max_prob >= 0.40:
        status_badge  = "🟡  MODERATE RISK"
        badge_color   = "#e65100"
        card_bg       = "#fff3e0"
        card_border   = "#e65100"
        action_text   = "🔧 Schedule maintenance soon"
    else:
        status_badge  = "🟢  SAFE"
        badge_color   = "#2e7d32"
        card_bg       = "#e8f5e9"
        card_border   = "#2e7d32"
        action_text   = "✅ No immediate action needed"

    # Detected failures
    detected = [FAILURE_FULL[i] for i, p in enumerate(preds) if p == 1]
    detected_text = ", ".join(detected) if detected else "None"

    status_md = f"""
<div style="
    background: {card_bg};
    border-left: 6px solid {card_border};
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.10);
">
    <h2 style="margin:0 0 10px 0; color:{badge_color}; font-size:1.7rem; font-weight:800;">{status_badge}</h2>
    <p style="margin:6px 0; font-size:1.08rem; color:#1a1a1a; font-weight:600;"><strong>Max Probability:</strong> {max_prob*100:.2f}%</p>
    <p style="margin:6px 0; font-size:1.08rem; color:#1a1a1a; font-weight:600;"><strong>Detected Failures:</strong> {detected_text}</p>
    <p style="margin:10px 0 0 0; font-size:1.0rem; color:#1a1a1a; font-weight:500;">{action_text}</p>
</div>
"""

    # ── Probability table ────────────────────────────────────────────────
    table_df = pd.DataFrame({
        "Failure Type": FAILURE_FULL,
        "Code":         FAILURE_LABELS,
        "Probability":  [f"{p*100:.2f}%" for p in probs],
        "Status":       ["⚠️ FAILURE" if pred == 1 else "✅ OK"
                         for pred in preds],
    })

    # ── Bar chart ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.8))

    # Color bars by risk
    bar_colors = []
    for p in probs:
        if p > 0.70:
            bar_colors.append("#ef4444")
        elif p >= 0.40:
            bar_colors.append("#f59e0b")
        else:
            bar_colors.append("#3b82f6")

    bars = ax.bar(FAILURE_LABELS, probs, color=bar_colors,
                  edgecolor="#1e293b", linewidth=0.8, width=0.55, zorder=3)

    # Value labels on top of each bar
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{p*100:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#1e293b")

    # Threshold lines
    ax.axhline(y=0.70, color="#ef4444", linestyle="--", linewidth=1, alpha=0.6, label="High Risk (0.70)")
    ax.axhline(y=0.40, color="#f59e0b", linestyle="--", linewidth=1, alpha=0.6, label="Moderate Risk (0.40)")

    ax.set_xlabel("Failure Type", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Probability", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Failure-Type Probabilities", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim(0, max(max(probs) + 0.12, 0.5))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#f8fafc")
    plt.tight_layout()

    return status_md, table_df, fig


# ─── Custom CSS ──────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Hide Gradio built-in footer & nav ─────────────── */
footer { display: none !important; }
.gradio-footer { display: none !important; }
.footer { display: none !important; }
.built-with { display: none !important; }
nav { display: none !important; }

/* ── Global ────────────────────────────────────────── */
.gradio-container {
    max-width: 1050px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* ── Title banner ──────────────────────────────────── */
#title-banner {
    text-align: center;
    padding: 28px 16px 10px;
}
#title-banner h1 {
    font-size: 2rem;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
#title-banner p {
    color: #64748b;
    font-size: 1.0rem;
    margin: 8px 0 0;
    max-width: 620px;
    margin-inline: auto;
}

/* ── Cards ─────────────────────────────────────────── */
.input-card, .output-card {
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 20px 24px !important;
    background: #ffffff !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.06) !important;
}
.dark .input-card, .dark .output-card {
    background: #1e293b !important;
    border-color: #334155 !important;
}

/* ── Predict button ────────────────────────────────── */
#predict-btn {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 12px 0 !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
#predict-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,.35) !important;
}

/* ── Slider tweaks ─────────────────────────────────── */
input[type="number"] {
    border-radius: 8px !important;
}

/* ── Table ──────────────────────────────────────────── */
table { border-radius: 10px !important; overflow: hidden; }
"""

# ─── Build the Gradio Blocks UI ──────────────────────────────────────────────

with gr.Blocks(
    title="MaintAI — Multi-Output Failure Predictor",
) as demo:

    # ── Title ────────────────────────────────────────────────────
    gr.HTML(
        """
        <div id="title-banner">
            <h1>🏭 MaintAI — Failure Predictor</h1>
            <p>
                Enter machine sensor readings below to predict five failure types
                simultaneously using a Multi-Output Random Forest classifier.
            </p>
        </div>
        """
    )

    with gr.Row(equal_height=True):

        # ── Left column — Inputs ─────────────────────────────────
        with gr.Column(scale=1, elem_classes="input-card"):
            gr.Markdown("### 🔧 Sensor Inputs")

            air_temp = gr.Slider(
                minimum=290, maximum=320, value=300, step=0.1,
                label="Air Temperature [K]",
                info="Ambient air temperature around the machine",
                interactive=True,
            )
            process_temp = gr.Slider(
                minimum=300, maximum=370, value=310, step=0.1,
                label="Process Temperature [K]",
                info="Internal process temperature",
                interactive=True,
            )
            rpm = gr.Slider(
                minimum=1000, maximum=3000, value=1500, step=1,
                label="Rotational Speed [rpm]",
                info="Spindle rotational speed",
                interactive=True,
            )
            torque = gr.Slider(
                minimum=0, maximum=100, value=40, step=0.1,
                label="Torque [Nm]",
                info="Torque applied to the tool",
                interactive=True,
            )
            tool_wear = gr.Slider(
                minimum=0, maximum=300, value=100, step=1,
                label="Tool Wear [min]",
                info="Accumulated tool usage time",
                interactive=True,
            )

            predict_btn = gr.Button(
                "  Predict Failures",
                variant="primary",
                elem_id="predict-btn",
            )

        # ── Right column — Outputs ───────────────────────────────
        with gr.Column(scale=1, elem_classes="output-card"):
            gr.Markdown("### <span style='color:#d32f2f; font-weight:800;'>📊 Prediction Results</span>")

            status_output = gr.HTML(label="Machine Status")
            table_output  = gr.Dataframe(
                label="Failure-Type Breakdown",
                headers=["Failure Type", "Code", "Probability", "Status"],
                interactive=False,
                wrap=True,
            )
            chart_output  = gr.Plot(label="Probability Chart")

    # ── Wire up ──────────────────────────────────────────────────
    predict_btn.click(
        fn=predict,
        inputs=[air_temp, process_temp, rpm, torque, tool_wear],
        outputs=[status_output, table_output, chart_output],
    )




# ─── Launch ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CUSTOM_CSS,
    )
