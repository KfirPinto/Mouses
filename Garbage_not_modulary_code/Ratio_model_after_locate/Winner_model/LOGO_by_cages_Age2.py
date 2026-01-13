import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import LeaveOneGroupOut
from LBL import LBL
import sys
import os

# --- 1. Suppress Warnings ---
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None 

# --- מחלקה לשמירת הפלט גם לקובץ וגם למסך ---
class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # נדרש עבור תאימות לפייתון
        self.terminal.flush()
        self.log.flush()

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
    # === הגדרת שמירת הלוגים ===
    output_dir = "/home/pintokf/Projects/Microbium/Mouses/Ratio_model_after_locate/winners"
    os.makedirs(output_dir, exist_ok=True) # יצירת התיקייה אם לא קיימת
    
    log_file_path = os.path.join(output_dir, "results_log.txt")
    
    # הפניית ההדפסות גם לקובץ
    sys.stdout = DualLogger(log_file_path)

    print(f"--- Log started. Saving to: {log_file_path} ---")
    print("--- Starting LOOCV Pipeline (Age 2 Months Only) ---")

    # --- 2. Load Data ---
    base_path = "/home/pintokf/Projects/Microbium/Mouses/Ratio_model_after_locate/preprocces_ratio_locate/"
    censored = pd.read_csv(base_path + "locate_censored_level_7.csv", index_col=0)
    uncensored = pd.read_csv(base_path + "locate_uncensored_level_7.csv", index_col=0)

    # --- 3. Clean Duplicates ---
    censored = censored[~censored.index.duplicated(keep='first')]
    uncensored = uncensored[~uncensored.index.duplicated(keep='first')]

    # --- 4. FILTER BY AGE (The Change) ---
    print(f"Total before filter -> Censored: {len(censored)}, Uncensored: {len(uncensored)}")
    
    censored = censored[censored["AgeMonths"] == 2]
    uncensored = uncensored[uncensored["AgeMonths"] == 2]
    
    print(f"Total AFTER filter (Age 2) -> Censored: {len(censored)}, Uncensored: {len(uncensored)}")

    # Extract Cage IDs
    censored["Cage"] = [i.split("-")[0] for i in censored.index]
    uncensored["Cage"] = [i.split("-")[0] for i in uncensored.index]

    print(f"Number of unique cages in Uncensored (Age 2): {uncensored['Cage'].nunique()}")

    if len(uncensored) < 2:
        print("ERROR: Not enough data for cross-validation after filtering!")
        exit()

    # --- 5. Initialize Leave-One-Group-Out ---
    logo = LeaveOneGroupOut()
    all_predictions_list = []

    print("\n--- Starting Cross-Validation Loop ---")
    
    # Split by CAGE
    for i, (train_idx, test_idx) in enumerate(logo.split(uncensored, groups=uncensored["Cage"])):
        
        # A. Split Data
        train_uncensored = uncensored.iloc[train_idx].copy()
        test_uncensored = uncensored.iloc[test_idx].copy()
        
        current_test_cage = test_uncensored["Cage"].iloc[0]
        print(f"Iteration {i+1}: Testing on Cage {current_test_cage} ({len(test_uncensored)} mice)")

        # === THE FIX: Drop Cage before fitting ===
        X_train = train_uncensored.drop(columns=["Cage"], errors='ignore')
        X_test = test_uncensored.drop(columns=["Cage"], errors='ignore')
        Y_censored = censored.drop(columns=["Cage"], errors='ignore')

        # B. Train Model (on filtered data)
        lbl = LBL("diff", "MiceName", "AgeMonths", num_of_bact=10, feature_selection=10, with_microbiome=True,
                  augmented_censored=False, gamma=0.0, only_microbiome=True, alpha=0.001)
        
        # Use clean dataframes
        lbl.fit(X_train, Y_censored)

        # C. Predict
        preds = lbl.predict(X_test)
        
        # D. Store results
        fold_results = test_uncensored[["diff"]].copy()
        fold_results["predicted_score"] = preds
        fold_results["Cage"] = current_test_cage
        all_predictions_list.append(fold_results)

    # --- 6. Aggregate & Score ---
    final_results = pd.concat(all_predictions_list)
    
    print("\n" + "="*40)
    print("FINAL AGGREGATED RESULTS (AGE 2)")
    print("="*40)
    print(final_results)

    y_true = final_results["diff"].values
    y_pred = final_results["predicted_score"].values

    spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    pearson_corr, pearson_p = pearsonr(y_true, y_pred)
    c_index = calculate_concordance_index(y_true, y_pred)

    print("\n--- Global Model Performance (Age 2) ---")
    print(f"Concordance Index (CI): {c_index:.4f}")
    print(f"Spearman Correlation:   {spearman_corr:.4f} (p-value: {spearman_p:.4g})")
    print(f"Pearson Correlation:    {pearson_corr:.4f} (p-value: {pearson_p:.4g})")

    # --- 7. Visualize ---
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='green', alpha=0.7, label='Age 2 Mice')
    plt.xlabel("True Survival Diff")
    plt.ylabel("Predicted Score")
    plt.title(f"LOOCV Results (Age 2 Only)\nCI: {c_index:.2f} | Spearman: {spearman_corr:.2f}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # שמירת הגרף גם בתיקיית ה-winners
    plot_path = os.path.join(output_dir, "LOOCV_age2_results_7.png")
    plt.savefig(plot_path)
    print(f"\nPlot saved to: {plot_path}")