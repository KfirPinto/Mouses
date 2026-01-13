# 1. Imports for suppression MUST come first
import warnings
import pandas as pd
import matplotlib.pyplot as plt

# 2. AGGRESSIVE WARNING SUPPRESSION
# These lines must run before importing LBL/MLE_augmentor
warnings.filterwarnings("ignore") 
pd.options.mode.chained_assignment = None  # This kills the specific SettingWithCopyWarning

# 3. Now import the library (it will inherit the settings above)
from sklearn.model_selection import GroupShuffleSplit
from LBL import LBL

if __name__ == '__main__':
    # 1. Load data (No filtering by age this time)
    base_path = "/home/pintokf/Projects/Microbium/Mouses/Ratio_model_after_locate/preprocces_ratio_locate/"
    censored = pd.read_csv(base_path + "locate_censored_level_7.csv", index_col=0)
    uncensored = pd.read_csv(base_path + "locate_uncensored_level_7.csv", index_col=0)

    # Extract Cage ID from the index (assuming format "Cage-MouseID_...")
    censored["Cage"] = [i.split("-")[0] for i in censored.index]
    uncensored["Cage"] = [i.split("-")[0] for i in uncensored.index]

    # Print total counts to verify we have all data
    print(f"Total Censored samples: {len(censored)}")
    print(f"Total Uncensored samples: {len(uncensored)}")

    # 2. Initialize the LBL model
    lbl = LBL("diff", "MiceName", "AgeMonths", num_of_bact=10, feature_selection=10, with_microbiome=True,
              augmented_censored=False, gamma=0.0, only_microbiome=True, alpha=0.001)

    # 3. Split Uncensored data into Train (70%) and Test (30%) by Cage
    # We use GroupShuffleSplit to ensure mice from the same cage are kept together
    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    
    # The loop runs only once because n_splits=1
    for train_idx, test_idx in gss.split(uncensored, groups=uncensored["Cage"]):
        train_uncensored = uncensored.iloc[train_idx]
        test_uncensored = uncensored.iloc[test_idx]

    # 4. Verify the split
    print("-" * 30)
    print("Split Verification:")
    print(f"Train set size (Uncensored): {len(train_uncensored)}")
    print(f"Test set size (Uncensored): {len(test_uncensored)}")
    
    # Check for overlapping cages
    train_cages = set(train_uncensored["Cage"].unique())
    test_cages = set(test_uncensored["Cage"].unique())
    overlap = train_cages.intersection(test_cages)
    
    if overlap:
        print(f"WARNING: Data leakage detected! Overlapping cages: {overlap}")
    else:
        print("SUCCESS: Clean split. No cage overlap between Train and Test.")

    # 5. Train the model
    # We use the training portion of Uncensored data AND ALL Censored data
    lbl.fit(train_uncensored, censored)

    # 6. Predict on the held-out Test set
    print("\n--- Predictions on Test Set ---")
    predictions = lbl.predict(test_uncensored)
    print(predictions)

    # 7. Calculate and print the score
    print("\n--- Model Score ---")
    try:
        score = lbl.score(test_uncensored, test_uncensored["diff"])
        print(score)
    except Exception as e:
        print(f"Error calculating score: {e}")

    # ... (After printing the score) ...

    # 8. Visualize the results
    # We want to see the correlation between True values (diff) and Predicted values
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(test_uncensored["diff"], predictions, color='blue', label='Test Samples')
    
    # Add labels and title
    plt.xlabel("True Survival Diff (Actual)")
    plt.ylabel("Predicted Score (Model)")
    plt.title(f"Prediction vs Actual\nCI: {score['ci']:.2f}, P-val: {score['pval']:.2f}")
    plt.grid(True, alpha=0.3)
    
    # Save the plot to a file so you can see it
    plt.savefig("ratio_model_results_7.png")
    print("\nPlot saved as 'ratio_model_results.png'")