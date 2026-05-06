"""
Stroke Prediction Analysis Pipeline
====================================
Module: Applied Data in Action (U19973)
Assessment 2 — Group Practical

Loads the Kaggle stroke prediction dataset, performs preprocessing
(handles missing values, removes duplicates, ensures consistency,
standardises scales), runs exploratory data analysis with
visualisations, fits a logistic regression and random forest, then
evaluates with metrics appropriate for an imbalanced classification
problem.

Dataset: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.impute import SimpleImputer
from scipy import stats

# ----------------- Visual style -----------------
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 110
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'
PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("1. LOAD DATA")
print("=" * 60)
df = pd.read_csv('/home/claude/stroke_project/healthcare-dataset-stroke-data.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)

# ============================================================
# 2. INITIAL EXPLORATION
# ============================================================
print("\n" + "=" * 60)
print("2. INITIAL EXPLORATION")
print("=" * 60)
print("\nMissing values per column:")
# bmi was loaded as object due to 'N/A' strings — convert to numeric
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
print(df.isnull().sum())

print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nClass balance for stroke:")
print(df['stroke'].value_counts(normalize=True))

# ============================================================
# 3. PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("3. PREPROCESSING")
print("=" * 60)

# 3.1 Drop the id column — not informative
df = df.drop(columns=['id'])

# 3.2 Handle the rare 'Other' gender (n=1) — drop to avoid dummy with one observation
n_other = (df['gender'] == 'Other').sum()
df = df[df['gender'] != 'Other'].copy()
print(f"Dropped {n_other} row(s) with gender='Other'")

# 3.3 Standardise categorical text — strip whitespace, ensure consistent capitalisation
text_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in text_cols:
    df[col] = df[col].astype(str).str.strip()

# Rename 'Residence_type' to lowercase for consistency
df = df.rename(columns={'Residence_type': 'residence_type'})

# 3.4 Handle missing bmi — median imputation (robust to outliers)
bmi_missing = df['bmi'].isnull().sum()
median_bmi = df['bmi'].median()
df['bmi'] = df['bmi'].fillna(median_bmi)
print(f"Imputed {bmi_missing} missing bmi values with median ({median_bmi:.1f})")

# 3.5 Drop duplicates (none expected, but defensive)
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows")

# 3.6 Encode binary categoricals as 0/1
df['ever_married'] = (df['ever_married'] == 'Yes').astype(int)
df['gender'] = (df['gender'] == 'Male').astype(int)  # 1 = Male, 0 = Female
df['residence_type'] = (df['residence_type'] == 'Urban').astype(int)

# 3.7 One-hot encode multi-level categoricals
df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=True)

print(f"\nFinal shape after preprocessing: {df.shape}")
print(f"Final columns: {list(df.columns)}")

# Save the cleaned data
df.to_csv('/home/claude/stroke_project/stroke_clean.csv', index=False)

# ============================================================
# 4. EXPLORATORY DATA ANALYSIS — VISUALISATIONS
# ============================================================
print("\n" + "=" * 60)
print("4. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# We'll need the original-format df for some plots — reload it
df_eda = pd.read_csv('/home/claude/stroke_project/healthcare-dataset-stroke-data.csv')
df_eda['bmi'] = pd.to_numeric(df_eda['bmi'], errors='coerce')
df_eda['bmi'] = df_eda['bmi'].fillna(df_eda['bmi'].median())
df_eda = df_eda[df_eda['gender'] != 'Other'].copy()

# ---- Figure 1: Class imbalance ----
fig, ax = plt.subplots(figsize=(7, 4.5))
counts = df_eda['stroke'].value_counts()
bars = ax.bar(['No Stroke (0)', 'Stroke (1)'], counts.values,
              color=[PALETTE[0], PALETTE[3]], edgecolor='black')
for bar, val in zip(bars, counts.values):
    pct = 100 * val / counts.sum()
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{val:,}\n({pct:.1f}%)', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Patients')
ax.set_title('Figure 1: Distribution of the Stroke Target Variable (Class Imbalance)')
ax.set_ylim(0, max(counts.values) * 1.15)
plt.tight_layout()
plt.savefig('/home/claude/stroke_project/fig1_class_imbalance.png', bbox_inches='tight')
plt.close()
print("Saved fig1_class_imbalance.png")

# ---- Figure 2: Age distribution by stroke status ----
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.histplot(data=df_eda, x='age', hue='stroke', bins=40, multiple='layer',
             palette=[PALETTE[0], PALETTE[3]], alpha=0.65, ax=ax)
ax.set_xlabel('Age (years)')
ax.set_ylabel('Number of Patients')
ax.set_title('Figure 2: Age Distribution by Stroke Status')
handles = ax.get_legend().legend_handles
ax.legend(handles, ['No Stroke', 'Stroke'], title='Outcome')
plt.tight_layout()
plt.savefig('/home/claude/stroke_project/fig2_age_distribution.png', bbox_inches='tight')
plt.close()
print("Saved fig2_age_distribution.png")

# ---- Figure 3: Stroke rate by key risk factors ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
risk_factors = ['hypertension', 'heart_disease', 'ever_married']
titles = ['Hypertension', 'Heart Disease', 'Ever Married']

for ax, factor, title in zip(axes, risk_factors, titles):
    if factor == 'ever_married':
        rates = df_eda.groupby(factor)['stroke'].mean() * 100
        labels = ['No', 'Yes']
    else:
        rates = df_eda.groupby(factor)['stroke'].mean() * 100
        labels = ['No', 'Yes']
    bars = ax.bar(labels, rates.values, color=[PALETTE[0], PALETTE[3]], edgecolor='black')
    for bar, val in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Stroke Rate (%)')
    ax.set_title(title)
    ax.set_ylim(0, max(rates.values) * 1.25)

plt.suptitle('Figure 3: Stroke Rate Across Key Binary Risk Factors', y=1.02)
plt.tight_layout()
plt.savefig('/home/claude/stroke_project/fig3_risk_factors.png', bbox_inches='tight')
plt.close()
print("Saved fig3_risk_factors.png")

# ---- Figure 4: Glucose vs BMI with stroke overlay ----
fig, ax = plt.subplots(figsize=(8, 5))
no_stroke = df_eda[df_eda['stroke'] == 0]
yes_stroke = df_eda[df_eda['stroke'] == 1]
ax.scatter(no_stroke['avg_glucose_level'], no_stroke['bmi'],
           alpha=0.25, s=14, color=PALETTE[0], label='No Stroke')
ax.scatter(yes_stroke['avg_glucose_level'], yes_stroke['bmi'],
           alpha=0.85, s=24, color=PALETTE[3], label='Stroke',
           edgecolor='black', linewidth=0.4)
ax.set_xlabel('Average Glucose Level (mg/dL)')
ax.set_ylabel('BMI')
ax.set_title('Figure 4: Glucose Level vs BMI, Coloured by Stroke Outcome')
ax.legend()
plt.tight_layout()
plt.savefig('/home/claude/stroke_project/fig4_glucose_bmi.png', bbox_inches='tight')
plt.close()
print("Saved fig4_glucose_bmi.png")

# ---- Figure 5: Correlation heatmap of numerical features ----
fig, ax = plt.subplots(figsize=(7, 5.5))
num_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']
corr = df_eda[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, ax=ax, cbar_kws={'label': 'Pearson r'})
ax.set_title('Figure 5: Correlation Heatmap of Numerical Variables')
plt.tight_layout()
plt.savefig('/home/claude/stroke_project/fig5_correlation.png', bbox_inches='tight')
plt.close()
print("Saved fig5_correlation.png")

# ---- Statistical tests ----
print("\n--- Statistical tests ---")
# Chi-square: hypertension vs stroke
ct = pd.crosstab(df_eda['hypertension'], df_eda['stroke'])
chi2, p, _, _ = stats.chi2_contingency(ct)
print(f"Hypertension vs Stroke: chi2={chi2:.2f}, p={p:.2e}")

ct = pd.crosstab(df_eda['heart_disease'], df_eda['stroke'])
chi2, p, _, _ = stats.chi2_contingency(ct)
print(f"Heart Disease vs Stroke: chi2={chi2:.2f}, p={p:.2e}")

# t-test: age between stroke groups
t, p = stats.ttest_ind(df_eda[df_eda['stroke']==1]['age'],
                       df_eda[df_eda['stroke']==0]['age'])
print(f"Age (stroke vs no-stroke): t={t:.2f}, p={p:.2e}")

t, p = stats.ttest_ind(df_eda[df_eda['stroke']==1]['avg_glucose_level'],
                       df_eda[df_eda['stroke']==0]['avg_glucose_level'])
print(f"Glucose (stroke vs no-stroke): t={t:.2f}, p={p:.2e}")

# ============================================================
# 5. MACHINE LEARNING — CLASSIFICATION
# ============================================================
print("\n" + "=" * 60)
print("5. MACHINE LEARNING")
print("=" * 60)

X = df.drop(columns=['stroke'])
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Standardise numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- Model 1: Logistic Regression with class weighting ----
print("\n--- Logistic Regression ---")
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
print(classification_report(y_test, y_pred_lr, target_names=['No Stroke', 'Stroke']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.3f}")

# ---- Model 2: Random Forest ----
print("\n--- Random Forest ---")
rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                            max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)  # tree models don't need scaling
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_rf, target_names=['No Stroke', 'Stroke']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.3f}")

# ---- Figure 6: Confusion matrices for both models ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, y_pred, name in zip(axes, [y_pred_lr, y_pred_rf],
                            ['Logistic Regression', 'Random Forest']):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['No Stroke', 'Stroke'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
    ax.set_title(name)
plt.suptitle('Figure 6: Confusion Matrices on Test Set (n=1022)', y=1.02)
plt.tight_layout()
plt.savefig('/home/claude/stroke_project/fig6_confusion.png', bbox_inches='tight')
plt.close()
print("Saved fig6_confusion.png")

# ---- Figure 7: ROC curves ----
fig, ax = plt.subplots(figsize=(6.5, 5))
for prob, name, colour in [(y_prob_lr, 'Logistic Regression', PALETTE[0]),
                           (y_prob_rf, 'Random Forest', PALETTE[1])]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, color=colour, lw=2, label=f'{name} (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random (AUC = 0.500)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate (Recall)')
ax.set_title('Figure 7: ROC Curves for Stroke Prediction Models')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('/home/claude/stroke_project/fig7_roc.png', bbox_inches='tight')
plt.close()
print("Saved fig7_roc.png")

# ---- Figure 8: Feature importance from Random Forest ----
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
colours = [PALETTE[3] if v > 0.05 else PALETTE[0] for v in importances.values]
ax.barh(range(len(importances)), importances.values, color=colours, edgecolor='black')
ax.set_yticks(range(len(importances)))
ax.set_yticklabels(importances.index)
ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)')
ax.set_title('Figure 8: Random Forest Feature Importances')
plt.tight_layout()
plt.savefig('/home/claude/stroke_project/fig8_feature_importance.png', bbox_inches='tight')
plt.close()
print("Saved fig8_feature_importance.png")

# Print summary statistics for the report
print("\n" + "=" * 60)
print("SUMMARY STATISTICS FOR REPORT")
print("=" * 60)
print(f"\nFinal cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Class balance: {y.mean()*100:.2f}% positive (stroke)")
print(f"\nLogistic Regression — Accuracy: {(y_pred_lr == y_test).mean():.3f}")
print(f"Logistic Regression — Recall (stroke): {(y_pred_lr[y_test==1] == 1).mean():.3f}")
print(f"Logistic Regression — ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.3f}")
print(f"\nRandom Forest       — Accuracy: {(y_pred_rf == y_test).mean():.3f}")
print(f"Random Forest       — Recall (stroke): {(y_pred_rf[y_test==1] == 1).mean():.3f}")
print(f"Random Forest       — ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.3f}")

print("\nTop 5 features by RF importance:")
print(importances.sort_values(ascending=False).head().to_string())
