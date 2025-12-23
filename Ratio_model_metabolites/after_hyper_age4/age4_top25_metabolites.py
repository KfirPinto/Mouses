import warnings
import pandas as pd
import numpy as np
import re
import sys
import os
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# --- 1. Suppress Warnings ---
warnings.filterwarnings("ignore")

# --- 2. Setup Logging ---
output_dir = "/home/pintokf/Projects/Microbium/Mouses/Ratio_model_metabolites/after_hyper_age4"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"--- Starting Top 25 Metabolite Extraction (Age 4) ---")

# --- 3. Load Data ---
base_path = "/home/pintokf/Projects/Microbium/Mouses/Ratio_model_metabolites/preprocces_ratio_metabolites/"
censored = pd.read_csv(base_path + "metabolites_censored.csv", index_col=0)
uncensored = pd.read_csv(base_path + "metabolites_uncensored.csv", index_col=0)

# --- Clean Column Names ---
def clean_col_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))

censored.columns = [clean_col_name(c) for c in censored.columns]
uncensored.columns = [clean_col_name(c) for c in uncensored.columns]

# --- 4. Filter for Age 4 Only ---
uncensored = uncensored[uncensored["AgeMonths"] == 4]
censored = censored[censored["AgeMonths"] == 4]

# Combine for broader context if needed, but usually we calculate correlations on Uncensored (known outcomes)
# Or if you want to impute censored, that's complex. Let's stick to Uncensored for accurate correlation with 'diff'
X_train_raw = uncensored.copy()

# --- CRITICAL FIX: Drop Metadata and Non-Numeric Columns ---
# We explicitly drop known metadata columns
cols_to_drop = ["diff", "AgeMonths", "MiceName", "Cage", "DeathDate", "DateOfBirth", "Gender", "Group"]
X_train = X_train_raw.drop(columns=cols_to_drop, errors='ignore')

# SAFETY NET: Keep ONLY numeric columns (float/int) to prevent crashes on strings
X_train = X_train.select_dtypes(include=[np.number])

y_train = X_train_raw["diff"]

print(f"Data Loaded: {X_train.shape[0]} samples (Age 4 Uncensored)")
print(f"Total numeric features available: {X_train.shape[1]}")

# --- 5. Step A: Find the Top 25 Features (Feature Selection) ---
correlations = []
for col in X_train.columns:
    try:
        # Calculate correlation ignoring NaNs
        # We ensure inputs are floats
        x_col = X_train[col].astype(float)
        coef, p = spearmanr(x_col, y_train)
        
        if not np.isnan(coef):
            correlations.append((col, abs(coef), coef)) 
    except Exception as e:
        # print(f"Skipping {col}: {e}")
        pass

# Create DataFrame of correlations
corr_df = pd.DataFrame(correlations, columns=["Metabolite", "Abs_Corr", "Real_Corr"])

# Sort by Absolute Correlation
top_25_df = corr_df.sort_values(by="Abs_Corr", ascending=False).head(25)
top_25_names = top_25_df["Metabolite"].tolist()

print(f"\nTop 25 Metabolites Selected:")
print(top_25_names)

# --- 6. Step B: Fit Ridge Regression on these 25 ---
# Prepare X with only top 25
X_subset = X_train[top_25_names]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_subset)

# Fit Ridge
ridge = Ridge(alpha=0.001)
ridge.fit(X_scaled, y_train)

# --- 7. Step C: Organize Results for Excel ---
results = []
for name, coeff in zip(top_25_names, ridge.coef_):
    # Interpretation logic
    if coeff > 0:
        direction = "Positive"
        meaning = "Pro-Survival (Higher levels = Live Longer)"
    else:
        direction = "Negative"
        meaning = "Risk Factor (Higher levels = Die Sooner)"
        
    results.append({
        "Metabolite_Name": name,
        "Ridge_Coefficient": coeff,
        "Direction": direction,
        "Biological_Interpretation": meaning
    })

results_df = pd.DataFrame(results)

# Sort by Magnitude of impact
results_df["Abs_Coeff"] = results_df["Ridge_Coefficient"].abs()
results_df = results_df.sort_values(by="Abs_Coeff", ascending=False).drop(columns=["Abs_Coeff"])

# --- 8. Save to CSV ---
output_file = os.path.join(output_dir, "age4_top25_biomarkers.csv")
results_df.to_csv(output_file, index=False)

print("\n" + "="*60)
print(f"SUCCESS! Biomarker list saved to: {output_file}")
print("="*60)
print(results_df.to_string())