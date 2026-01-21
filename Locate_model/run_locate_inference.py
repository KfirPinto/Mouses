import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import LOCATE
import os

# === Settings ===
base_path = "/home/pintokf/Projects/Microbium/Mouses"
# Inputs
micro_path = f"{base_path}/MIPMLP_scripts/whole_metadata/processed_subpca_level7.csv"
metabo_path = f"{base_path}/preprocess_metabolits/preprocessed_metabolites_normalized_z_score.csv"
# Outputs
output_dir = f"{base_path}/Locate_model/Whole_data/inference"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def load_data_raw():
    """Load raw data without filtering, to serve as the final inference set."""
    try:
        df_micro = pd.read_csv(micro_path)
        if 'ID' in df_micro.columns: df_micro.set_index('ID', inplace=True)
        return df_micro
    except Exception as e:
        print(f"❌ Error loading raw microbiome: {e}")
        exit(1)

def load_and_align_data():
    print("--- 1. Loading Data for Training ---")
    # Load Microbiome
    try:
        df_micro = pd.read_csv(micro_path)
        if 'ID' in df_micro.columns:
            df_micro.set_index('ID', inplace=True)
        print(f"Microbiome loaded: {df_micro.shape} (Samples, Features)")
    except Exception as e:
        print(f"❌ Error loading microbiome: {e}")
        exit(1)

    # Load Metabolites
    try:
        df_metabo = pd.read_csv(metabo_path)
        if 'SampleID' in df_metabo.columns:
            print("Notice: Renaming 'SampleID' to 'ID' in Metabolites file.")
            df_metabo.rename(columns={'SampleID': 'ID'}, inplace=True)
        
        if 'ID' in df_metabo.columns:
            df_metabo.set_index('ID', inplace=True)
        print(f"Metabolites loaded: {df_metabo.shape} (Samples, Features)")
    except Exception as e:
        print(f"❌ Error loading metabolites: {e}")
        exit(1)

    print("\n--- 2. Aligning Data (Intersection) ---")
    # Find common IDs (The Intersection) - ONLY for training!
    common_ids = df_micro.index.intersection(df_metabo.index)
    
    if len(common_ids) == 0:
        print("❌ CRITICAL ERROR: No common IDs found!")
        exit(1)
        
    print(f"✅ Found {len(common_ids)} common samples for TRAINING.")
    
    # Filter for training
    X = df_micro.loc[common_ids]
    Y = df_metabo.loc[common_ids]
    
    return X, Y

def save_z(matrix, index, filename):
    # 1. Create DataFrame
    if isinstance(matrix, pd.DataFrame):
        df_z = matrix.copy()
        df_z.index = index
    else:
        df_z = pd.DataFrame(matrix, index=index)
        
    # 2. Reset index
    df_z.reset_index(inplace=True)
    if 'index' in df_z.columns: df_z.rename(columns={'index': 'ID'}, inplace=True)
    elif df_z.columns[0] != 'ID': df_z.rename(columns={df_z.columns[0]: 'ID'}, inplace=True)

    # 3. Rename Z columns
    num_feature_cols = df_z.shape[1] - 1
    feature_names = [f'Z_{i}' for i in range(num_feature_cols)]
    df_z.columns = ['ID'] + feature_names
    
    # 4. Save
    path = f"{output_dir}/{filename}"
    df_z.to_csv(path, index=False)
    print(f"✅ Saved: {path} (Shape: {df_z.shape})")


if __name__ == '__main__':
    # 1. Get Aligned Data (Only samples that have both Micro + Metabo)
    X_paired, Y_paired = load_and_align_data()

    # 2. Split (Train/Validation)
    print("\n--- 3. Splitting Train/Test (on paired data) ---")
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_paired, Y_paired, test_size=0.2, random_state=42
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Val set:   {X_val.shape[0]} samples")

    # 3. Train LOCATE
    print("\n--- 4. Training LOCATE ---")
    model = LOCATE.LOCATE_training(X_train, Y_train, X_val, Y_val)

    # === חלק א': שמירת ה-Train וה-Test המקוריים (כמו שביקשת) ===
    print("\n--- 5a. Saving Paired Train/Test Z Files ---")
    
    # חיזוי ל-Train
    Z_train_matrix, _ = LOCATE.LOCATE_predict(model, X_train, Y_train.columns)
    save_z(Z_train_matrix, X_train.index, "locate_Z_train_level_7.csv")

    # חיזוי ל-Test
    Z_val_matrix, _ = LOCATE.LOCATE_predict(model, X_val, Y_val.columns)
    save_z(Z_val_matrix, X_val.index, "locate_Z_test_level_7.csv")


    # === חלק ב': הוספת העכברים שאין להם מטבוליטים (Inference) ===
    print("\n--- 5b. Inferring Z for WHOLE cohort (including unpaired) ---")
    
    # א. טוענים את כל המיקרוביום המקורי
    df_micro_all = load_data_raw()
    print(f"Total Microbiome samples to predict: {df_micro_all.shape[0]}")
    
    # ב. מייצרים Z לכולם
    Z_all_matrix, _ = LOCATE.LOCATE_predict(model, df_micro_all, Y_train.columns)

    # ג. שמירה של הקובץ המלא
    print("\n--- 6. Saving Final Whole Cohort Z File ---")
    save_z(Z_all_matrix, df_micro_all.index, "locate_Z_whole_cohort_level_7.csv")