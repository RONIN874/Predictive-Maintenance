# 🏭 MaintAI — Multi-Output Failure Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?logo=scikit-learn&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-6.x-yellow?logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A machine-learning system that predicts **five industrial failure types simultaneously** using real-time sensor data and a **Multi-Output Random Forest classifier**, with a modern **Gradio web UI** for interactive predictions.

---

## 📋 Project Overview

Unplanned machine failures in industrial settings lead to costly downtime and safety risks. This project builds a **multi-output predictive maintenance model** that analyses sensor readings — temperature, rotational speed, torque, and tool wear — to classify which specific failure types are likely to occur.

- **Algorithm**: `MultiOutputClassifier(RandomForestClassifier)` with balanced class weights
- **Dataset**: [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) — UCI Machine Learning Repository
- **Task**: Multi-output binary classification (5 failure types predicted simultaneously)
- **UI**: Gradio Blocks web interface for interactive predictions

---

## 🎯 Prediction Targets

Instead of a single binary "Machine failure" output, the model predicts **five failure types** at once:

| Code | Failure Type | Description |
|---|---|---|
| **TWF** | Tool Wear Failure | Failure due to excessive tool wear |
| **HDF** | Heat Dissipation Failure | Failure due to overheating |
| **PWF** | Power Failure | Failure due to power issues |
| **OSF** | Overstrain Failure | Failure due to overload / overstrain |
| **RNF** | Random Failure | Random / unexplained failure |

### Risk Level Rules

| Condition | Status |
|---|---|
| Max probability > 70% | 🔴 **HIGH RISK** — Immediate maintenance required |
| Max probability ≥ 40% | 🟡 **MODERATE RISK** — Schedule maintenance soon |
| Max probability < 40% | 🟢 **SAFE** — No immediate action needed |

---

## 🔧 Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Core language |
| pandas | 1.5+ | Data loading & manipulation |
| NumPy | 1.23+ | Numerical operations |
| scikit-learn | 1.2+ | Multi-output classifier, preprocessing, evaluation |
| matplotlib | 3.6+ | Plotting & visualisation |
| seaborn | 0.12+ | Statistical plots (heatmaps, box plots) |
| Gradio | 6.x | Interactive web UI for predictions |

---

## 📊 Dataset Description

**Source**: AI4I 2020 Predictive Maintenance Dataset  
**Records**: 10,000 | **Columns**: 14

### Input Features (X) — 5 columns

| Feature | Description | Unit |
|---|---|---|
| Air temperature | Air temperature around the machine | K |
| Process temperature | Internal process temperature | K |
| Rotational speed | Spindle rotational speed | rpm |
| Torque | Torque applied by the machine | Nm |
| Tool wear | Cumulative tool wear time | min |

### Targets (Y) — 5 columns

| Target | Description |
|---|---|
| TWF | Tool Wear Failure (0 or 1) |
| HDF | Heat Dissipation Failure (0 or 1) |
| PWF | Power Failure (0 or 1) |
| OSF | Overstrain Failure (0 or 1) |
| RNF | Random Failure (0 or 1) |


---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/RONIN874/Predictive-Maintenance.git
cd Predictive-Maintenance

# Install dependencies
pip install -r requirements.txt
pip install gradio
```

---

## ⚙️ Usage

### 🌐 Launch the Gradio Web UI

Start the interactive prediction interface:

```bash
python app.py
```

Open your browser at `http://127.0.0.1:7860` and use the sliders to input sensor values. Click **Predict Failures** to see:

- Machine status (SAFE / MODERATE RISK / HIGH RISK)
- Per-failure-type probabilities
- A bar chart of all five failure probabilities

### 🏋️ Train the Model

Runs the full pipeline: data loading → preprocessing → EDA → training → evaluation → model saving.

```bash
python main.py train
```

### 🔮 Run CLI Predictions

**Interactive mode** — prompts for each parameter:

```bash
python main.py predict
```

**With CLI arguments**:

```bash
python main.py predict --air-temp 298.1 --process-temp 308.6 --rpm 1551 --torque 42.8 --tool-wear 0
```

**Example output**:

```
========================================
  Multi-Output Failure Prediction
========================================

Input Parameters:
  Air Temperature    : 298.1 K
  Process Temperature: 308.6 K
  Rotational Speed   : 1551 rpm
  Torque             : 42.8 Nm
  Tool Wear          : 0 min

✅ Machine Status: Safe (No failures predicted)
```

### 📊 Run EDA Only

Generate exploratory data analysis plots without training:

```bash
python main.py eda
```

---

## 📁 Project Structure

```
Predictive-Maintenance/
│
├── data/                          # Raw dataset
│   └── ai4i2020.csv
│
├── models/                        # Saved trained models & preprocessors
│   ├── random_forest_model.pkl    # MultiOutputClassifier(RandomForest)
│   └── scaler.pkl                 # StandardScaler
│
├── notebooks/                     # Jupyter notebooks for exploration
│   └── exploration.ipynb
│
├── outputs/                       # Generated plots and reports
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── failure_distribution.png
│   ├── failure_boxplots.png
│   └── feature_importance.png
│
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data loading, cleaning, scaling
│   ├── eda.py                     # Exploratory data analysis plots
│   ├── model_training.py          # Multi-output model training & evaluation
│   ├── model_utils.py             # Save / load model utilities
│   └── predict.py                 # Multi-output prediction logic
│
├── app.py                         # Gradio web UI for predictions
├── main.py                        # CLI entry point (train / predict / eda)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

---

## 📈 Results

### Model Performance

The Multi-Output Random Forest classifier predicts all five failure types simultaneously with balanced class weights to handle the imbalanced dataset.

### Feature Importance

![Feature Importance](outputs/feature_importance.png)

---

## 📊 EDA Highlights

### Feature Distributions

![Feature Distributions](outputs/feature_distributions.png)

### Correlation Heatmap

![Correlation Heatmap](outputs/correlation_heatmap.png)

### Failure Type Distribution

![Failure Distribution](outputs/failure_distribution.png)

### Feature Box Plots by Failure Status

![Failure Box Plots](outputs/failure_boxplots.png)

## 🙏 Acknowledgements

- **Dataset**: [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) — UCI Machine Learning Repository
- **Tools**: scikit-learn, pandas, matplotlib, seaborn, Gradio
