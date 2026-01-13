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
        
    log_file_path = os.path.join(output_dir, "simple_split_10_times_feature_selection_results.txt")
    sys.stdout = Logger(log_file_path)

    print(f"--- Starting Repeated Random Split Sweep (10 repeats per k) ---")
    print(f"Log saved to: {log_file_path}")

    # --- 5. Load Data ---
    base_path = "/home/pintokf/Projects/Microbium/Mouses/Ratio_model_metabolites/preprocces_ratio_metabolites/"
    censored_path = base_path + "metabolites_censored.csv"
    uncensored_path = base_path + "metabolites_uncensored.csv"

    censored = pd.read_csv(censored_path, index_col=0)
    uncensored = pd.read_csv(uncensored_path, index_col=0)

    # --- Clean Column Names ---
    def clean_col_name(name):
        return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))

    censored.columns = [clean_col_name(c) for c in censored.columns]
    uncensored.columns = [clean_col_name(c) for c in uncensored.columns]

    # Extract Cage IDs
    censored["Cage"] = [i.split("-")[0] for i in censored.index]
    uncensored["Cage"] = [i.split("-")[0] for i in uncensored.index]

    # Note: Using ALL data (You can uncomment the next lines to filter by Age 4 if needed)
    # uncensored = uncensored[uncensored["AgeMonths"] == 4]
    # censored = censored[censored["AgeMonths"] == 4]

    print(f"Total Censored samples: {len(censored)}")
    print(f"Total Uncensored samples: {len(uncensored)}")

    # --- 6. Define Feature Selection Values ---
    k_values_to_test = [10, 25, 50, 100, 250, 500, 1000, 1889]
    N_REPEATS = 10  # How many random splits to run for each k
    
    summary_results = []

    # --- 7. The Sweep Loop ---
    for k in k_values_to_test:
        print(f"\n" + "#"*60)
        print(f">>> Testing Top {k} features (Running {N_REPEATS} splits) <<<")
        print("#"*60)
        
        # Lists to store scores for the 10 repeats
        ci_scores = []
        spearman_scores = []
        pearson_scores = []
        
        # Initialize Splitter
        # Setting random_state=42 ensures that the 10 splits are identical across different k values
        # (Fair comparison: k=10 and k=25 are tested on the exact same 10 subsets)
        gss = GroupShuffleSplit(n_splits=N_REPEATS, train_size=0.7, random_state=42)
        
        fold_idx = 1
        for train_idx, test_idx in gss.split(uncensored, groups=uncensored["Cage"]):
            train_uncensored = uncensored.iloc[train_idx]
            test_uncensored = uncensored.iloc[test_idx]
            
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
                # Train
                lbl.fit(train_uncensored.copy(), censored.copy())
                
                # Predict
                preds = lbl.predict(test_uncensored.copy())
                
                # Metrics
                y_true = test_uncensored["diff"].values
                ci = calculate_concordance_index(y_true, preds)
                spr_corr, _ = spearmanr(y_true, preds)
                pear_corr, _ = pearsonr(y_true, preds)
                
                ci_scores.append(ci)
                spearman_scores.append(spr_corr)
                pearson_scores.append(pear_corr)
                
                # Optional: print dot to show progress
                print(f"   Split {fold_idx}/{N_REPEATS}: CI={ci:.4f}", end="\r")
                fold_idx += 1
                
            except Exception as e:
                print(f"   Split {fold_idx} Failed: {e}")

        # --- Aggregate Results for this k ---
        mean_ci = np.mean(ci_scores)
        std_ci = np.std(ci_scores)
        mean_spr = np.mean(spearman_scores)
        mean_pear = np.mean(pearson_scores)
        
        print(f"\n   >>> Average for k={k}: CI={mean_ci:.4f} (+/- {std_ci:.4f})")
        
        summary_results.append({
            "Features_Selected": k,
            "Mean_CI": mean_ci,
            "Std_CI": std_ci,
            "Mean_Spearman": mean_spr,
            "Mean_Pearson": mean_pear
        })

    # --- 8. Final Summary ---
    print("\n" + "="*80)
    print(f"FINAL AGGREGATED RESULTS ({N_REPEATS} Random 70/30 Splits per k)")
    print("="*80)
    
    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        df_summary = df_summary.sort_values(by="Mean_CI", ascending=False)
        print(df_summary.to_string(index=False))
        
        best_k = df_summary.iloc[0]["Features_Selected"]
        print(f"\nBest average configuration: {best_k} features.")
    else:
        print("No results generated.")