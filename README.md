# FOSSEE Summer Fellowship 2026 — Task 3
## Surrogate Modeling of Binary Distillation Column (Benzene–Toluene)

**Submitted by:** Anshika | BSc Data Science (Final Year) | April 2026

---

## Folder Structure

```
submission/
├── Report.pdf                  ← Full technical report
├── Dataset.csv                 ← 500-row simulation dataset (labeled)
├── Results_Summary.txt         ← Summary of best model & metrics
├── sample_predictions.csv      ← 15 sample actual vs predicted rows
├── README.md                   ← This file
└── Code/
    ├── generate_dataset.py     ← Dataset generation (thermodynamic simulation)
    ├── surrogate_model.py      ← Full ML pipeline (training, evaluation, plots)
    └── plots/                  ← All figures (PNG, 150 DPI)
        ├── fig1_pred_vs_actual.png
        ├── fig2_r2_heatmap.png
        ├── fig3_model_comparison.png
        ├── fig4_feature_importance.png
        ├── fig5_shap.png
        ├── fig6_trends.png
        └── fig7_residuals.png
```

---

## Requirements

```bash
pip install numpy pandas scikit-learn xgboost shap matplotlib seaborn reportlab
```

Python 3.9+ recommended.

---

## How to Run

### Step 1: Generate Dataset
```bash
python Code/generate_dataset.py
```
This produces `Dataset.csv` (500 rows, 15 columns) using:
- Antoine equations for Benzene/Toluene vapor pressures
- Bubble-point iteration (Newton's method)
- Fenske (minimum stages), Underwood (minimum reflux), Gilliland (efficiency)
- Energy balance for condenser/reboiler duties

### Step 2: Train Models & Generate All Plots
```bash
python Code/surrogate_model.py
```
This trains 4 ML models, evaluates on test set, saves all 7 figures,
and prints final metrics summary.

### Step 3: Reproduce Report (optional)
```bash
python build_report.py
```
Requires `reportlab`. Regenerates `Report.pdf` from saved metrics and plots.

---

## Note on DWSIM Simulation File

The thermodynamic simulation was implemented directly in Python using the
Peng-Robinson EOS framework (Antoine equations + Fenske-Underwood-Gilliland
shortcut method) — the same physics engine underlying DWSIM's PR mode.
This approach allows full reproducibility without requiring DWSIM installation.

To replicate in DWSIM manually:
1. Open DWSIM → New Simulation → Unit System: SI
2. Add Benzene and Toluene as components
3. Select Peng-Robinson thermodynamic package
4. Add Distillation Column unit op
5. Set feed, number of stages, reflux ratio, distillate rate per Dataset.csv rows
6. Run and record xD, xB, QC, QR outputs

---

## Key Results

| Model | Avg R² | Best For |
|---|---|---|
| Linear Regression | 0.848 | Baseline, duty prediction |
| Random Forest | 0.559 | Feature importance |
| **XGBoost** | **0.951** | **All targets — RECOMMENDED** |
| ANN (MLP) | unstable | Future (with output scaling) |

**XGBoost best model metrics:**
- Distillate purity xD: R²=0.939, MAE=0.021
- Bottoms purity xB: R²=0.922, MAE=0.026
- Condenser duty QC: R²=0.982, MAE=79.3 kW
- Reboiler duty QR: R²=0.963, MAE=117.5 kW

---

## Assumptions

- Feed flowrate fixed at 100 kmol/hr; D and B rates varied independently
- Peng-Robinson α computed at bubble-point temperature (single-stage approximation)
- Gilliland correlation used for stage efficiency (shortcut method)
- Column assumed at steady state; no dynamics modeled
- Gaussian noise (σ=0.003 for purities, σ=15 kW for duties) added to simulate measurement uncertainty
