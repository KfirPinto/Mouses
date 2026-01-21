import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import LOCATE
import os

# === Settings ===
base_path = "/home/pintokf/Projects/Microbium/Mouses"
# Inputs
micro_path = f"{base_path}/MIPMLP_scripts/whole_metadata/processed_subpca_level6.csv"
metabo_path = f"{base_path}/preprocess_metabolits/preprocessed_metabolites_normalized_z_score.csv"
# Outputs
output_dir = f"{base_path}/Locate_model/Whole_data"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def load_and_align_data():
    print("--- 1. Loading Data ---")
    # Load Microbiome
    try:
        df_micro = pd.read_csv(micro_path)
        # Ensure ID is the index for proper joining
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
        
        # Ensure ID is the index
        if 'ID' in df_metabo.columns:
            df_metabo.set_index('ID', inplace=True)
        print(f"Metabolites loaded: {df_metabo.shape} (Samples, Features)")
    except Exception as e:
        print(f"❌ Error loading metabolites: {e}")
        exit(1)

    print("\n--- 2. Aligning Data (Intersection) ---")
    # Find common IDs (The Intersection)
    common_ids = df_micro.index.intersection(df_metabo.index)
    
    if len(common_ids) == 0:
        print("❌ CRITICAL ERROR: No common IDs found between Microbiome and Metabolites!")
        print("Check if one file uses 'ID_1' and the other '1' (string vs int format).")
        exit(1)
        
    print(f"✅ Found {len(common_ids)} common samples (Mice present in both files).")
    
    # Filter both dataframes to keep only the common IDs, in the SAME order
    X = df_micro.loc[common_ids]
    Y = df_metabo.loc[common_ids]
    
    return X, Y

def save_z(matrix, index, filename):
    """
    Robust function to save Z matrix. 
    It calculates column names based on the actual dataframe size 
    to prevent Length Mismatch errors.
    """
    # 1. Create DataFrame
    if isinstance(matrix, pd.DataFrame):
        df_z = matrix.copy()
        df_z.index = index
    else:
        df_z = pd.DataFrame(matrix, index=index)
        
    # 2. Reset index to move ID into the columns
    # The index (ID) becomes the first column (column 0)
    df_z.reset_index(inplace=True)
    
    # 3. Generate Column Names
    # We force the first column to be 'ID'
    # We name the rest Z_0, Z_1... based on how many columns exist
    
    num_feature_cols = df_z.shape[1] - 1  # Total columns minus the ID column
    feature_names = [f'Z_{i}' for i in range(num_feature_cols)]
    
    # Assign new names
    df_z.columns = ['ID'] + feature_names
    
    # 4. Save
    path = f"{output_dir}/{filename}"
    df_z.to_csv(path, index=False)
    print(f"✅ Saved: {path} (Shape: {df_z.shape})")


if __name__ == '__main__':
    # 1. Get Aligned Data
    X, Y = load_and_align_data()

    # 2. Split Together
    print("\n--- 3. Splitting Train/Test ---")
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_val.shape[0]} samples")

    # 3. Train LOCATE
    print("\n--- 4. Training LOCATE ---")
    model = LOCATE.LOCATE_training(X_train, Y_train, X_val, Y_val)

    # 4. Extract Z (Latent Representation)
    print("\n--- 5. Extracting Z Features ---")
    
    # Extract
    Z_val_matrix, _ = LOCATE.LOCATE_predict(model, X_val, Y_val.columns)
    Z_train_matrix, _ = LOCATE.LOCATE_predict(model, X_train, Y_train.columns)

    # 5. Save Results
    print("\n--- 6. Saving Z Files for RATIO ---")
    
    # Use the new robust save function
    save_z(Z_train_matrix, X_train.index, "locate_Z_train_level_6.csv")
    save_z(Z_val_matrix, X_val.index, "locate_Z_test_level_6.csv")