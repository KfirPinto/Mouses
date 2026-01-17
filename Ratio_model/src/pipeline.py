import pandas as pd
import os
import sys
from sklearn.model_selection import LeaveOneGroupOut

# Import LBL from ratio-t2e package (installed via pip)
sys.path.insert(0, '/home/pintokf/miniconda3/envs/ratio_env/lib/python3.10/site-packages')
from LBL import LBL

from src.evaluation import evaluate_and_plot

# מחלקת לוגר כדי לשמור את הפלטים לקובץ טקסט
class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def create_output_dir(cfg):
    base = cfg['output_settings']['base_folder']
    group = cfg['output_settings']['experiment_group']
    name = cfg['output_settings']['model_name']
    full_path = os.path.join(base, group, name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path

def run_logo_cv(censored, uncensored, params, feature_k):
    """
    מריץ סיבוב LOOCV אחד.
    מקבל את כל הפרמטרים מה-YAML ומעביר אותם ל-LBL.
    """
    logo = LeaveOneGroupOut()
    all_predictions = []

    # שימוש בפרמטרים מתוך הקונפיגורציה
    lbl_params = {
        "tag_column": params['target_col'],
        "id_column": params['id_col'],
        "order_of_samples_column": params['age_col'],
        "num_of_bact": params['num_of_bact'],
        "feature_selection": feature_k,  # דינמי
        "with_microbiome": params['with_microbiome'],
        "augmented_censored": params['augmented_censored'],
        "gamma": params['gamma'],
        "only_microbiome": params['only_microbiome'],
        "alpha": params['alpha']
    }
    
    # הוספת categories אם קיים (למקרה שצריך בעתיד)
    if 'categories' in params:
        lbl_params['categories'] = params['categories']

    for i, (train_idx, test_idx) in enumerate(logo.split(uncensored, groups=uncensored["Cage"])):
        train = uncensored.iloc[train_idx]
        test = uncensored.iloc[test_idx]
        current_cage = test["Cage"].iloc[0]

        # אתחול המודל עם כל הפרמטרים
        lbl = LBL(**lbl_params)
        
        try:
            lbl.fit(train.copy(), censored.copy())
            preds = lbl.predict(test.copy())
            
            fold_res = test[[params['target_col']]].copy()
            fold_res["predicted_score"] = preds
            fold_res["Cage"] = current_cage
            all_predictions.append(fold_res)
            
        except Exception as e:
            import traceback
            print(f"Error in Cage {current_cage}: {e}")
            traceback.print_exc()

    if not all_predictions:
        return None

    return pd.concat(all_predictions)

def run_pipeline(cfg, run_hyper=False):
    output_dir = create_output_dir(cfg)
    censored, uncensored = cfg['data_loaded']
    
    # הפניית ההדפסות גם לקובץ לוג
    sys.stdout = Logger(os.path.join(output_dir, "run_log.txt"))
    
    print(f">>> Output Directory: {output_dir}")
    print(f">>> Model Configuration: {cfg['model_params']}")

    if run_hyper:
        print("\n>>> MODE: Hyperparameter Search <<<")
        k_values = cfg['hyperparameters']['k_values']
        summary = []
        
        for k in k_values:
            print(f"\n--- Testing feature_selection k={k} ---")
            results_df = run_logo_cv(censored, uncensored, cfg['model_params'], k)
            
            if results_df is not None:
                metrics = evaluate_and_plot(results_df, output_dir, file_prefix=f"results_k{k}")
                metrics['k'] = k
                summary.append(metrics)
        
        # שמירת סיכום
        if summary:
            summary_df = pd.DataFrame(summary).sort_values(by="c_index", ascending=False)
            summary_path = os.path.join(output_dir, "hyper_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print("\n=== Hyperparameter Search Results ===")
            print(summary_df)
        
    else:
        print("\n>>> MODE: Single Run <<<")
        k = cfg['model_params']['feature_selection']
        print(f"Running with k={k}")
        
        results_df = run_logo_cv(censored, uncensored, cfg['model_params'], k)
        
        if results_df is not None:
            evaluate_and_plot(results_df, output_dir, file_prefix="final_results")
            results_df.to_csv(os.path.join(output_dir, "predictions.csv"))
            print("Done.")