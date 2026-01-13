import warnings
import pandas as pd
import numpy as np
import re
import sys
import os
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import LeaveOneGroupOut
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
    output_dir = "/home/pintokf/Projects/Microbium/Mouses/Ratio_model_microbium/hyper_feature_selection_level_6"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    log_file_path = os.path.join(output_dir, "feature_selection_all_age_results.txt")
    sys.stdout = Logger(log_file_path)

    print(f"--- Starting Feature Selection Sweep (Detailed Output) ---")
    print(f"Log saved to: {log_file_path}")

    # --- 5. Load Data ---
    base_path = "/home/pintokf/Projects/Microbium/Mouses/Ratio_model_microbium/Preprocces_for_ratio_model/"
    censored = pd.read_csv(base_path + "data_level6_censored.csv", index_col=0)
    uncensored = pd.read_csv(base_path + "data_level6_uncensored.csv", index_col=0)

    # Filtering Age 4 (Efficiently done before splitting cages)
    #uncensored = uncensored[uncensored["AgeMonths"] == 4]
    #censored = censored[censored["AgeMonths"] == 4]

    # --- Clean Column Names ---
    def clean_col_name(name):
        return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))

    censored.columns = [clean_col_name(c) for c in censored.columns]
    uncensored.columns = [clean_col_name(c) for c in uncensored.columns]

    # Extract Cage IDs
    censored["Cage"] = [i.split("-")[0] for i in censored.index]
    uncensored["Cage"] = [i.split("-")[0] for i in uncensored.index]

    print(f"Total Censored samples : {len(censored)}")
    print(f"Total Uncensored samples : {len(uncensored)}")

    # --- 6. Define Feature Selection Values ---
    k_values_to_test = [5, 10, 15, 20, 25, 30, 35]
    summary_results = []

    # --- 7. The Sweep Loop ---
    for k in k_values_to_test:
        print(f"\n" + "#"*60)
        print(f">>> Testing Top {k} features <<<")
        print("#"*60)
        
        logo = LeaveOneGroupOut()
        all_predictions_list = []
        
        for i, (train_idx, test_idx) in enumerate(logo.split(uncensored, groups=uncensored["Cage"])):
            train_uncensored = uncensored.iloc[train_idx]
            test_uncensored = uncensored.iloc[test_idx]
            current_cage = test_uncensored["Cage"].iloc[0]
            
            # --- Init Model ---
            lbl = LBL(
                "diff",          
                "MiceName",      
                "AgeMonths",     
                num_of_bact=35,         
                feature_selection=k,      
                with_microbiome=True,
                augmented_censored=False, 
                gamma=0.0, 
                only_microbiome=True, 
                alpha=0.001,
                categories=[]
            )
            
            try:
                lbl.fit(train_uncensored.copy(), censored.copy())
                preds = lbl.predict(test_uncensored.copy())
                
                # --- Create Result DataFrame for this Fold ---
                fold_results = test_uncensored[["diff"]].copy()
                fold_results["predicted_score"] = preds
                fold_results["Cage"] = current_cage
                
                # --- PRINT DETAILED PREDICTIONS FOR THIS MOUSE/CAGE ---
                print(f"\n   [Cage {current_cage}] Predictions:")
                print(fold_results.to_string(header=False)) 
                
                all_predictions_list.append(fold_results)
                
            except Exception as e:
                print(f"Error in Cage {current_cage}: {e}")

        # --- Aggregate Results for this k ---
        if len(all_predictions_list) > 0:
            final_results = pd.concat(all_predictions_list)
            y_true = final_results["diff"].values
            y_pred = final_results["predicted_score"].values

            # Calculate Scores
            ci = calculate_concordance_index(y_true, y_pred)
            spr_corr, spr_p = spearmanr(y_true, y_pred)
            
            # --- ADDED: Pearson Correlation Calculation ---
            pear_corr, pear_p = pearsonr(y_true, y_pred)
            
            print(f"\n   >>> Summary for k={k}: CI={ci:.4f} | Spearman={spr_corr:.4f} | Pearson={pear_corr:.4f}")
            
            # --- ADDED: Updated Summary Dictionary ---
            summary_results.append({
                "Features_Selected": k,
                "CI": ci,
                "Spearman_Corr": spr_corr,
                "Pearson_Corr": pear_corr,
                "P_Value_Spearman": spr_p
            })
        else:
            print(f"   [Result for k={k}] Failed to run.")

    # --- 8. Final Summary ---
    print("\n" + "="*50)
    print("FINAL COMPARISON TABLE")
    print("="*50)
    
    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        df_summary = df_summary.sort_values(by="CI", ascending=False)
        print(df_summary.to_string(index=False))
        
        best_k = df_summary.iloc[0]["Features_Selected"]
        print(f"\nBest configuration: {best_k} features.")
    else:
        print("No results generated.")