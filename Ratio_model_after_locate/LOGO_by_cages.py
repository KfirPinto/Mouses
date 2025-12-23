import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import LeaveOneGroupOut
from LBL import LBL

# --- 1. Suppress Warnings ---
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None 

def calculate_concordance_index(y_true, y_pred):
    """
    Helper function to calculate C-Index manually for the aggregated results.
    Counts pairs where the model correctly predicted the order.
    """
    n = len(y_true)
    assert n == len(y_pred)
    
    count = 0
    correct = 0
    
    # Compare every pair of samples
    for i in range(n):
        for j in range(i + 1, n):
            # We only compare if the true values are different
            if y_true[i] != y_true[j]:
                count += 1
                # Check if model predicted the same order/direction as truth
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                   (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    correct += 1
                # Handle ties in prediction (0.5 score)
                elif y_pred[i] == y_pred[j]:
                    correct += 0.5
                    
    return correct / count if count > 0 else 0.5

if __name__ == '__main__':
    print("--- Starting Leave-One-Cage-Out (LOGO) Pipeline ---")

    # --- 2. Load Data ---
    # Using the files defined in your path
    base_path = "/home/pintokf/Projects/Microbium/Mouses/Ratio_model_after_locate/preprocces_ratio_locate/"
    censored = pd.read_csv(base_path + "locate_censored_level_7.csv", index_col=0)
    uncensored = pd.read_csv(base_path + "locate_uncensored_level_7.csv", index_col=0)

    # Extract Cage IDs
    censored["Cage"] = [i.split("-")[0] for i in censored.index]
    uncensored["Cage"] = [i.split("-")[0] for i in uncensored.index]

    # Optional: Filter by age if you want to test specifically on Age 2
    # uncensored = uncensored[uncensored["AgeMonths"] == 2]
    # censored = censored[censored["AgeMonths"] == 2] 

    print(f"Total Censored samples: {len(censored)}")
    print(f"Total Uncensored samples: {len(uncensored)}")
    print(f"Number of unique cages in Uncensored: {uncensored['Cage'].nunique()}")

    # --- 3. Initialize Leave-One-Group-Out ---
    logo = LeaveOneGroupOut()
    
    # We will store the results here
    # This list will hold small DataFrames, each containing the predictions for one cage
    all_predictions_list = []

    print("\n--- Starting Cross-Validation Loop ---")
    
    # Iterate through each split
    # groups=uncensored["Cage"] ensures we split by CAGE, not by mouse
    for i, (train_idx, test_idx) in enumerate(logo.split(uncensored, groups=uncensored["Cage"])):
        
        # A. Split Data
        train_uncensored = uncensored.iloc[train_idx]
        test_uncensored = uncensored.iloc[test_idx]
        
        current_test_cage = test_uncensored["Cage"].iloc[0]
        print(f"Iteration {i+1}: Testing on Cage {current_test_cage} ({len(test_uncensored)} mice)")

        # B. Initialize & Train Model
        # We re-initialize the model each time to ensure no data leakage from previous folds
        lbl = LBL("diff", "MiceName", "AgeMonths", num_of_bact=10, feature_selection=10, with_microbiome=True,
                  augmented_censored=False, gamma=0.0, only_microbiome=True, alpha=0.001)
        
        # Train on: (All other cages) + (All Censored data)
        lbl.fit(train_uncensored.copy(), censored.copy())
        # C. Predict on the kept-out cage
        preds = lbl.predict(test_uncensored)
        
        # D. Store results
        # Create a temporary DataFrame with True values and Predicted values
        fold_results = test_uncensored[["diff"]].copy()
        fold_results["predicted_score"] = preds
        fold_results["Cage"] = current_test_cage
        
        all_predictions_list.append(fold_results)

    # --- 4. Aggregate Results ---
    # Combine all the small DataFrames into one big result table
    final_results = pd.concat(all_predictions_list)
    
    print("\n" + "="*40)
    print("FINAL AGGREGATED RESULTS")
    print("="*40)
    print(final_results)

    # --- 5. Calculate Global Score ---
    y_true = final_results["diff"].values
    y_pred = final_results["predicted_score"].values

    # Calculate Metrics
    # Spearman: Good for ranking (non-linear relationships)
    spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    # Pearson: Good for linear relationships
    pearson_corr, pearson_p = pearsonr(y_true, y_pred)
    # C-Index: The metric used in your previous outputs
    c_index = calculate_concordance_index(y_true, y_pred)

    print("\n--- Global Model Performance ---")
    print(f"Concordance Index (CI): {c_index:.4f}")
    print(f"Spearman Correlation:   {spearman_corr:.4f} (p-value: {spearman_p:.4g})")
    print(f"Pearson Correlation:    {pearson_corr:.4f} (p-value: {pearson_p:.4g})")

    # --- 6. Visualize ---
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, color='purple', alpha=0.7, label='Mice')
    
    # Add labels
    plt.xlabel("True Survival Diff (Actual)")
    plt.ylabel("Predicted Score (LOOCV)")
    plt.title(f"Leave-One-Cage-Out Validation\nCI: {c_index:.2f}, Spearman r: {spearman_corr:.2f}")
    plt.grid(True, alpha=0.3)
    
    # Add identity line for reference (optional)
    # plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', alpha=0.5)

    output_plot = "LOOCV_results_7.png"
    plt.savefig(output_plot)
    print(f"\nPlot saved as '{output_plot}'")