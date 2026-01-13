import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import os

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

def evaluate_and_plot(results_df, output_dir, file_prefix="results"):
    y_true = results_df["diff"].values
    y_pred = results_df["predicted_score"].values

    c_index = calculate_concordance_index(y_true, y_pred)
    spearman_corr, sp_p = spearmanr(y_true, y_pred)
    pearson_corr, pe_p = pearsonr(y_true, y_pred)

    print(f"\nResults for {file_prefix}:")
    print(f"  C-Index: {c_index:.4f}")
    print(f"  Spearman: {spearman_corr:.4f} (p={sp_p:.4g})")
    print(f"  Pearson: {pearson_corr:.4f} (p={pe_p:.4g})")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, color='purple', alpha=0.7)
    plt.xlabel("True Survival Diff")
    plt.ylabel("Predicted Score (LOOCV)")
    plt.title(f"LOOCV Prediction\nCI: {c_index:.2f}, Spearman: {spearman_corr:.2f}")
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, f"{file_prefix}_plot.png")
    plt.savefig(plot_path)
    plt.close()

    return {
        "c_index": c_index, 
        "spearman": spearman_corr, 
        "pearson": pearson_corr, 
        "p_spearman": sp_p
    }