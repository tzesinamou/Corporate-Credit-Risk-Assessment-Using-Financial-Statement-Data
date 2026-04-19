import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# =========================
# PATHS
# =========================
base_path = "financial_csv"
output_path = "."

# =========================
# METRICS MAP
# =========================
REQUIRED_METRICS = [
    'Total Assets', 'Total Debt', 'Stockholders Equity',
    'Current Assets', 'Current Liabilities',
    'Net Income', 'EBITDA', 'Total Revenue',
    'Operating Cash Flow', 'Free Cash Flow'
]

def normalize_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip().lower().replace(" ", "").replace("_", "")

MAPPING = {normalize_text(m): m for m in REQUIRED_METRICS}

# =========================
# CLEAN NUMERIC
# =========================
def clean_numeric(df):
    df = df.copy()

    for col in df.columns:
        if col not in ["Metric", "Company", "Year"]:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace("(", "-", regex=False)
                .str.replace(")", "", regex=False)
                .str.strip()
            )

            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# =========================
# LOAD DATA
# =========================
all_data = []
excluded_companies = []

for company in os.listdir(base_path):
    company_path = os.path.join(base_path, company)

    if not os.path.isdir(company_path):
        continue

    frames = []
    found_files = []

    for file in ["BalanceSheet.csv", "CashFlow.csv", "IncomeStatement.csv"]:
        file_path = os.path.join(company_path, file)

        if not os.path.exists(file_path):
            continue

        try:
            df = pd.read_csv(file_path)

            if df.empty:
                continue

            df.rename(columns={df.columns[0]: "Metric"}, inplace=True)

            new_cols = ["Metric"]
            for c in df.columns[1:]:
                try:
                    new_cols.append(str(pd.to_datetime(c).year))
                except:
                    new_cols.append(str(c))

            df.columns = new_cols

            df["Metric"] = df["Metric"].apply(
                lambda x: MAPPING.get(normalize_text(x), None)
            )

            df = df.dropna(subset=["Metric"])

            years = ['2025', '2024', '2023', '2022']
            df = df[['Metric'] + [y for y in years if y in df.columns]]

            df = df.melt(id_vars="Metric", var_name="Year", value_name="Value")

            frames.append(df)
            found_files.append(file)

        except Exception as e:
            print(f"Error in {company}: {e}")

    if len(found_files) < 3:
        excluded_companies.append(company)
        continue

    if frames:
        comp = pd.concat(frames)
        comp["Company"] = company

        comp = clean_numeric(comp)

        pivot = comp.pivot_table(
            index=["Company", "Year"],
            columns="Metric",
            values="Value",
            aggfunc="mean"
        ).reset_index()

        all_data.append(pivot)

# =========================
# MERGE
# =========================
final_df = pd.concat(all_data, ignore_index=True)

final_df = final_df.replace([np.inf, -np.inf], np.nan)

# fill numeric safely
num_cols = final_df.select_dtypes(include=[np.number]).columns
final_df[num_cols] = final_df[num_cols].fillna(final_df[num_cols].median())

# =========================
# FEATURE ENGINEERING
# =========================
def safe_div(a, b):
    return np.where((b == 0) | (pd.isna(b)), np.nan, a / b)

final_df['Debt_to_Equity'] = safe_div(final_df['Total Debt'], final_df['Stockholders Equity'])
final_df['Current_Ratio'] = safe_div(final_df['Current Assets'], final_df['Current Liabilities'])
final_df['ROA'] = safe_div(final_df['Net Income'], final_df['Total Assets'])
final_df['EBITDA_Margin'] = safe_div(final_df['EBITDA'], final_df['Total Revenue'])
final_df['FCF_to_Debt'] = safe_div(final_df['Free Cash Flow'], final_df['Total Debt'])

# =========================
# LABEL
# =========================
risk_score = (
    (final_df['Debt_to_Equity'] > 2).astype(int) +
    (final_df['ROA'] < 0.02).astype(int) +
    (final_df['FCF_to_Debt'] < 0).astype(int) +
    (final_df['Current_Ratio'] < 1).astype(int)
)

final_df['Risk'] = (risk_score >= 2).astype(int)

# =========================
# RESTORE BUSINESS COLUMNS
# =========================

def risk_level(x):
    if x == 0:
        return "Low"
    elif x == 1:
        return "Medium"
    else:
        return "High"

final_df["Risk_Level"] = risk_score.apply(risk_level)

final_df["Performance_Index"] = (
    final_df["ROA"] * 0.4 +
    final_df["EBITDA_Margin"] * 0.3 +
    final_df["Current_Ratio"] * 0.2 +
    final_df["FCF_to_Debt"] * 0.1
)

min_val = final_df["Performance_Index"].min()
max_val = final_df["Performance_Index"].max()

final_df["Performance_Index"] = (
    (final_df["Performance_Index"] - min_val) /
    (max_val - min_val + 1e-9)
) * 100

final_df["Financial_Performance"] = np.where(
    final_df["ROA"] > 0.05, "Strong",
    np.where(final_df["ROA"] > 0.02, "Medium", "Weak")
)

final_df["Decision"] = np.where(
    (final_df["Risk"] == 0) &
    (final_df["Performance_Index"] >= 60) &
    (final_df["Financial_Performance"] != "Weak"),
    "Invest",
    "Do Not Invest"
)

print(final_df['Risk'].value_counts())

# =========================
# FEATURES (Using Raw Metrics to avoid Data Leakage)
# =========================
features = [
    'Total Assets', 'Total Debt', 'Stockholders Equity',
    'Current Assets', 'Current Liabilities',
    'Net Income', 'EBITDA', 'Total Revenue',
    'Operating Cash Flow', 'Free Cash Flow'
]

final_df = final_df.dropna(subset=features + ['Risk'])

X = final_df[features]
y = final_df['Risk']

# =========================
# SPLIT (Train - Validation - Test)
# =========================
# First split: 80% for dev (train+val), 20% for final test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: From the 80%, take 25% for validation (which is 20% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =========================
# MODELS (Trained on X_train)
# =========================
log_model = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train)
rf_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
xgb_model = xgb.XGBClassifier(eval_metric='logloss').fit(X_train, y_train)

# =========================
# EVALUATION FUNCTION (Updated for Validation)
# =========================
def evaluate(model, X_data, y_data, name):
    y_pred = model.predict(X_data)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_data)[:, 1]
    else:
        y_prob = y_pred

    acc = accuracy_score(y_data, y_pred)
    print(f"\n{name} (Validation)")
    print("Accuracy:", acc)

    cm = confusion_matrix(y_data, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(name + " Confusion Matrix (Val)")
    plt.show()

    fpr, tpr, _ = roc_curve(y_data, y_prob)
    return fpr, tpr, auc(fpr, tpr), acc

# =========================
# VALIDATION COMPARISON
# =========================
plt.figure()

for model, data, name in [
    (log_model, X_val_scaled, "Logistic Regression"),
    (rf_model, X_val, "Random Forest"),
    (xgb_model, X_val, "XGBoost")
]:
    fpr, tpr, roc_auc, acc = evaluate(model, data, y_val, name)
    plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc:.2f}")

plt.plot([0, 1], [0, 1], "--")
plt.legend()
plt.title("ROC Comparison (Validation Set)")
plt.show()

# =========================
# EXCEL EXPORT
# =========================
final_file = os.path.join(output_path, "FINAL_OUTPUT.xlsx")
df_export = final_df.copy()
wb = Workbook()
ws = wb.active
ws.title = "Data"
for r in dataframe_to_rows(df_export, index=False, header=True):
    ws.append(r)
for row in ws.iter_rows(min_row=2):
    for cell in row:
        if isinstance(cell.value, (int, float)) and cell.value is not None:
            cell.number_format = '#,##0.00'
wb.save(final_file)

# =========================
# FINAL REPORT (ON TEST SET)
# =========================
print("\n========================")
print("FINAL RESULTS (UNSEEN TEST SET)")
print("========================")

for name, model, data in [
    ("Logistic Regression", log_model, X_test_scaled),
    ("Random Forest", rf_model, X_test),
    ("XGBoost", xgb_model, X_test)
]:
    acc = accuracy_score(y_test, model.predict(data))
    print(name, ":", round(acc, 3))

print("\nSaved file:", final_file)
print("Excluded companies:", excluded_companies)