import warnings
import pandas as pd
import numpy as np
import re
import sys
import os
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import GroupShuffleSplit
from LBL import LBL

# --- 1. Suppress Warnings ---
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None 

# --- 2. Logger Class ---
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- 3. Helper Function for C-Index ---
def calculate_concordance_index(y_true, y_pred):
    n = len(y_true)
    count = 0
    correct = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                count += 1
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                   (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    correct += 1
                elif y_pred[i] == y_pred[j]:
                    correct += 0.5
    return correct / count if count > 0 else 0.5

if __name__ == '__main__':
    # --- 4. Setup Logging ---
    output_dir = "/home/pintokf/Projects/Microbium/Mouses/Ratio_model_metabolites/hyper_feature_selection"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    log_file_path = os.path.join(output_dir, "simple_split_feature_selection_results.txt")
    sys.stdout = Logger(log_file_path)

    print(f"--- Starting Feature Selection Sweep (Simple 70/30 Split) ---")
    print(f"Log saved to: {log_file_path}")

    # --- 5. Load Data ---
    base_path = "/home/pintokf/Projects/Microbium/Mouses/Ratio_model_metabolites/preprocces_ratio_metabolites/"
    censored_path = base_path + "metabolites_censored.csv"
    uncensored_path = base_path + "metabolites_uncensored.csv"

    censored = pd.read_csv(censored_path, index_col=0)
    uncensored = pd.read_csv(uncensored_path, index_col=0)

    # --- Clean Column Names (Mandatory) ---
    def clean_col_name(name):
        return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))

    censored.columns = [clean_col_name(c) for c in censored.columns]
    uncensored.columns = [clean_col_name(c) for c in uncensored.columns]

    # Extract Cage IDs
    censored["Cage"] = [i.split("-")[0] for i in censored.index]
    uncensored["Cage"] = [i.split("-")[0] for i in uncensored.index]

    # Note: Using ALL data (No age filtering) as requested in the snippet
    print(f"Total Censored samples: {len(censored)}")
    print(f"Total Uncensored samples: {len(uncensored)}")

    # --- 6. Perform Split ONCE (Consistency Check) ---
    # We split here so all feature selection runs use the EXACT SAME train/test set
    print("-" * 30)
    print("Preparing Fixed 70/30 Split...")
    
    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    
    train_idx, test_idx = next(gss.split(uncensored, groups=uncensored["Cage"]))
    
    fixed_train_uncensored = uncensored.iloc[train_idx]
    fixed_test_uncensored = uncensored.iloc[test_idx]
    
    print(f"Train Size: {len(fixed_train_uncensored)}")
    print(f"Test Size:  {len(fixed_test_uncensored)}")
    
    # Check overlap
    train_cages = set(fixed_train_uncensored["Cage"].unique())
    test_cages = set(fixed_test_uncensored["Cage"].unique())
    if train_cages.intersection(test_cages):
        print("CRITICAL WARNING: Cage leakage detected!")
    else:
        print("Split is clean (No cage overlap).")

    # --- 7. Define Feature Selection Values ---
    k_values_to_test = [10, 25, 50, 100, 250, 500, 1000, 1889]
    summary_results = []

    # --- 8. The Sweep Loop ---
    for k in k_values_to_test:
        print(f"\n" + "#"*60)
        print(f">>> Testing Top {k} features <<<")
        print("#"*60)
        
        # Init Model
        lbl = LBL(
            "diff",          
            "MiceName",      
            "AgeMonths",     
            num_of_bact=1889,         
            feature_selection=k,      
            with_microbiome=True,
            augmented_censored=False, 
            gamma=0.0, 
            only_microbiome=True, 
            alpha=0.001,
            categories=[]
        )
        
        try:
            # Train using the FIXED split + All Censored
            lbl.fit(fixed_train_uncensored.copy(), censored.copy())
            
            # Predict
            preds = lbl.predict(fixed_test_uncensored.copy())
            
            # --- Create Result DataFrame ---
            results_df = fixed_test_uncensored[["diff", "Cage"]].copy()
            results_df["predicted_score"] = preds
            
            # --- PRINT DETAILED TABLE ---
            print("\n   [Detailed Predictions on Test Set]")
            print(results_df.to_string())
            
            # --- Calculate Scores ---
            y_true = results_df["diff"].values
            y_pred = results_df["predicted_score"].values

            ci = calculate_concordance_index(y_true, y_pred)
            spr_corr, spr_p = spearmanr(y_true, y_pred)
            pear_corr, pear_p = pearsonr(y_true, y_pred)
            
            print(f"\n   >>> Summary for k={k}:")
            print(f"   CI: {ci:.4f}")
            print(f"   Spearman: {spr_corr:.4f} (p={spr_p:.4g})")
            print(f"   Pearson:  {pear_corr:.4f} (p={pear_p:.4g})")
            
            summary_results.append({
                "Features_Selected": k,
                "CI": ci,
                "Spearman_Corr": spr_corr,
                "Pearson_Corr": pear_corr,
                "P_Value_Spearman": spr_p
            })
            
        except Exception as e:
            print(f"Error executing k={k}: {e}")

    # --- 9. Final Summary ---
    print("\n" + "="*60)
    print("FINAL COMPARISON TABLE (Sorted by CI)")
    print("="*60)
    
    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        df_summary = df_summary.sort_values(by="CI", ascending=False)
        print(df_summary.to_string(index=False))
        
        best_k = df_summary.iloc[0]["Features_Selected"]
        print(f"\nBest configuration for this split: {best_k} features.")
    else:
        print("No results generated.")