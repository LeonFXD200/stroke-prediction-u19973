# Stroke Prediction — U19973 Applied Data in Action (Assessment 2)

This repository contains the code and analysis for the group practical assessment on the Kaggle Stroke Prediction Dataset.

## Group members
- Leon Dupree
- Chester Speed

## Dataset
The dataset is **not** included in this repository (terms of use). Download it from Kaggle:
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Save the file as `healthcare-dataset-stroke-data.csv` in the repository root.

## Files
- `stroke_analysis.ipynb` — main Jupyter notebook (runs end-to-end on Google Colab)
- `analysis.py` — same pipeline as a standalone Python script
- `Assignment2_Stroke_Prediction.docx` — final report

## How to run
1. Download the CSV from the Kaggle link above and place it in the project root.
2. Open `stroke_analysis.ipynb` in Google Colab or run `python analysis.py` locally.
3. Required packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`.

## Pipeline summary
1. **Load** the Kaggle CSV (5,110 patients × 12 features).
2. **Preprocess** — convert `N/A` BMI strings to numeric, impute missing BMI with the median, drop the single `Other` gender record, encode binary categoricals as 0/1, one-hot encode `work_type` and `smoking_status`, and standardise numerical features for the regression model.
3. **EDA** — class balance, age distribution by stroke status, stroke rates by binary risk factors, glucose × BMI scatter, and a numerical correlation heatmap.
4. **Statistical tests** — chi-square for categorical vs stroke; Welch's t-test for continuous vs stroke.
5. **Modelling** — logistic regression with `class_weight='balanced'` and a 300-tree random forest, both evaluated with classification report, confusion matrix and ROC-AUC on a stratified 80/20 split.

## Key findings
- Severe class imbalance: only 4.9% of patients are stroke-positive.
- Age is the dominant predictor (~40% of random forest feature importance), followed by glucose, BMI, hypertension and heart disease.
- Logistic regression (with class weighting) catches **70% of strokes** at 77% accuracy; the random forest looks "more accurate" at 91% but only catches **12% of strokes** — a textbook accuracy paradox.
- ROC-AUC: 0.79 (logistic regression) vs 0.75 (random forest).

## Module
Applied Data in Action (U19973), Level 6, Canterbury Christ Church University.
