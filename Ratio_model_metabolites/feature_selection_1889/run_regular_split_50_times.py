import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from LBL import LBL

# --- 1. Aggressive Warning Suppression ---
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None 

if __name__ == '__main__':
    print("--- Starting Repeated 70/30 Split Evaluation (50 Iterations) ---")

    # --- 2. Load Data (Level 7 as requested) ---
    # נתיבים לקבצים שלך
    censored_path = "/home/pintokf/Projects/Microbium/Mouses/Ratio_model_metabolites/preprocces_ratio_metabolites/metabolites_censored.csv"
    uncensored_path = "/home/pintokf/Projects/Microbium/Mouses/Ratio_model_metabolites/preprocces_ratio_metabolites/metabolites_uncensored.csv"
    
    censored = pd.read_csv(censored_path, index_col=0)
    uncensored = pd.read_csv(uncensored_path, index_col=0)

    # חילוץ הכלובים
    censored["Cage"] = [i.split("-")[0] for i in censored.index]
    uncensored["Cage"] = [i.split("-")[0] for i in uncensored.index]

    # ניקוי כפילויות אם יש (ליתר ביטחון)
    censored = censored[~censored.index.duplicated(keep='first')]
    uncensored = uncensored[~uncensored.index.duplicated(keep='first')]

    print(f"Total Censored: {len(censored)}")
    print(f"Total Uncensored: {len(uncensored)}")

    # --- 3. Setup Repeated Split ---
    n_iterations = 50
    # n_splits=50 אומר לו לייצר 50 חלוקות שונות ואקראיות
    gss = GroupShuffleSplit(n_splits=n_iterations, train_size=0.7, random_state=42)
    
    # רשימות לשמירת התוצאות
    ci_scores = []
    p_values = []
    correlations = []

    print(f"\nRunning {n_iterations} iterations...")
    print("-" * 50)

    # --- 4. The Loop ---
    # הפונקציה split מייצרת בכל איטרציה אינדקסים חדשים לחלוקה
    for i, (train_idx, test_idx) in enumerate(gss.split(uncensored, groups=uncensored["Cage"])):
        
        # A. Split Data
        train_uncensored = uncensored.iloc[train_idx]
        test_uncensored = uncensored.iloc[test_idx]
        
        # B. Initialize Model (New instance every time)
        lbl = LBL("diff", "MiceName", "AgeMonths", num_of_bact=35, feature_selection=35, with_microbiome=True,
                  augmented_censored=False, gamma=0.0, only_microbiome=True, alpha=0.001)

        # C. Train (Using .copy() to prevent data corruption!)
        lbl.fit(train_uncensored.copy(), censored.copy())

        # D. Evaluate
        try:
            score = lbl.score(test_uncensored, test_uncensored["diff"])
            
            # שמירת התוצאות
            ci_scores.append(score['ci'])
            if score['pval'] is not None:
                p_values.append(score['pval'])
            if score['corr'] is not None:
                correlations.append(score['corr'])
                
            # הדפסה קצרה לכל 10 ריצות כדי שתדע שזה עובד
            if (i + 1) % 5 == 0:
                print(f"Iteration {i+1}/{n_iterations} -> CI: {score['ci']:.3f}")
                
        except Exception as e:
            print(f"Iteration {i+1} Failed: {e}")

    # --- 5. Final Report ---
    print("-" * 50)
    print("FINAL RESULTS (Average of 50 runs)")
    print("-" * 50)
    
    mean_ci = np.mean(ci_scores)
    std_ci = np.std(ci_scores)
    
    mean_pval = np.mean(p_values) if p_values else float('nan')
    mean_corr = np.mean(correlations) if correlations else float('nan')

    print(f"Concordance Index (CI): {mean_ci:.4f} (+/- {std_ci:.4f})")
    print(f"Average Correlation:    {mean_corr:.4f}")
    print(f"Average P-value:        {mean_pval:.4f}")
    
    print("\nInterpretation:")
    if mean_ci > 0.5:
        print("-> The model performs better than random guessing on average.")
    else:
        print("-> The model performs worse or equal to random guessing on average.")