import warnings
import pandas as pd
import numpy as np
import re
import os
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# --- 1. Settings ---
warnings.filterwarnings("ignore")

# Paths
base_path = "/home/pintokf/Projects/Microbium/Mouses/Preprocess_ratio/preprocces_ratio_metabolites/"
output_dir = "/home/pintokf/Projects/Microbium/Mouses/results/Metabolites/Winner_For_Metabolites"

# Configuration (Standard Winner for Metabolites: k=25)
NUM_FEATURES = 25
DATA_FILE = "metabolites_uncensored.csv"

print(f"--- Starting Metabolites Coefficient Extraction (Whole Data, Top {NUM_FEATURES}) ---")

# --- 2. Load Data ---
uncensored = pd.read_csv(os.path.join(base_path, DATA_FILE), index_col=0)

# Clean Column Names
def clean_col_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))

uncensored.columns = [clean_col_name(c) for c in uncensored.columns]

# --- 3. Prepare X and y ---
X_train_raw = uncensored.copy()
cols_to_drop = ["diff", "AgeMonths", "MiceName", "Cage", "DeathDate", "DateOfBirth", "Gender", "Group"]
X_train = X_train_raw.drop(columns=cols_to_drop, errors='ignore')

# Keep only numeric
X_train = X_train.select_dtypes(include=[np.number])
y_train = X_train_raw["diff"]

print(f"Data Loaded: {X_train.shape[0]} samples")

# --- 4. Feature Selection (Spearman) ---
correlations = []
for col in X_train.columns:
    try:
        x_col = X_train[col].astype(float)
        coef, p = spearmanr(x_col, y_train)
        if not np.isnan(coef):
            correlations.append((col, abs(coef), coef)) 
    except:
        pass

corr_df = pd.DataFrame(correlations, columns=["Feature", "Abs_Corr", "Real_Corr"])
top_features_df = corr_df.sort_values(by="Abs_Corr", ascending=False).head(NUM_FEATURES)
top_feature_names = top_features_df["Feature"].tolist()

print(f"\nTop {NUM_FEATURES} Metabolites Selected:")
print(top_feature_names)

# --- 5. Ridge Regression ---
X_subset = X_train[top_feature_names]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_subset)

# Fit Ridge (alpha=0.001 matches LBL default)
ridge = Ridge(alpha=0.001)
ridge.fit(X_scaled, y_train)

# --- 6. Results ---
results = []
for name, coeff in zip(top_feature_names, ridge.coef_):
    if coeff > 0:
        direction = "Positive"
        meaning = "Pro-Survival"
    else:
        direction = "Negative"
        meaning = "Risk Factor"
        
    results.append({
        "Metabolite": name,
        "Ridge_Coefficient": coeff,
        "Direction": direction,
        "Interpretation": meaning
    })

results_df = pd.DataFrame(results).sort_values(by="Ridge_Coefficient", key=abs, ascending=False)
output_file = os.path.join(output_dir, f"metabolites_top{NUM_FEATURES}_coeffs.csv")
results_df.to_csv(output_file, index=False)
print(f"\nSaved to: {output_file}\n")
print(results_df.to_string())
