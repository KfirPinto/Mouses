import pandas as pd
import MIPMLP
import os

# === Settings ===
input_path = "/home/pintokf/Projects/Microbium/Mouses/Union_tables_To_MIPMLP/for_preprocess.csv"
output_dir = "/home/pintokf/Projects/Microbium/Mouses/MIPMLP_scripts"

# === Helper function for proper saving with ID ===
def save_with_id(df_result, filename):
    """
    Receives the MIPMLP result, ensures ID exists as a column, and saves it.
    """
    # 1. Handle case where a Tuple is returned (happens in some versions)
    if isinstance(df_result, tuple):
        df_result = df_result[0]
    
    # 2. Reset index to turn the ID into a standard column
    #    (Otherwise to_csv with index=False would delete it)
    df_result = df_result.reset_index()
    
    # 3. Ensure the column name is ID
    #    Usually after reset_index the column is named 'index' or as it was originally
    if 'index' in df_result.columns:
        df_result.rename(columns={'index': 'ID'}, inplace=True)
    elif df_result.columns[0] != 'ID':
        # If the first column is not ID, rename it to ID just to be safe
        df_result.rename(columns={df_result.columns[0]: 'ID'}, inplace=True)

    # 4. Save
    out_path = f"{output_dir}/{filename}"
    df_result.to_csv(out_path, index=False)
    print(f"✅ Saved to: {out_path}")
    print(f"   Shape: {df_result.shape}")
    print(f"   First column: {df_result.columns[0]}") # Final check

# === Start of execution ===
print(f"Loading data from: {input_path}")
try:
    # Loading - we keep the ID as index during the loading phase
    # MIPMLP knows how to work when ID is the index
    df = pd.read_csv(input_path)
    print(f"Data loaded. Shape: {df.shape}")
    
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# ==========================================
# Run 1: Level 7 (Species)
# ==========================================
print("\n--- Processing Level 7 (Species) ---")
try:
    processed_L7 = MIPMLP.preprocess(
        df,
        taxonomy_level=7,
        taxnomy_group='sub PCA',      # Note: 'mean' was selected in your code   
        )
    
    # Use the corrected function for saving
    save_with_id(processed_L7, "processed_subpca_level7.csv")

except Exception as e:
    print(f"❌ Error in Level 7: {e}")
    import traceback
    traceback.print_exc()

# ==========================================
# Run 2: Level 6 (Genus)
# ==========================================
print("\n--- Processing Level 6 (Genus) ---")
try:
    processed_L6 = MIPMLP.preprocess(
        df,
        taxonomy_level=6,
        taxnomy_group='sub PCA',      # Note: 'mean' was selected in your code
    )

    # Use the corrected function for saving
    save_with_id(processed_L6, "processed_subpca_level6.csv")

except Exception as e:
    print(f"❌ Error in Level 6: {e}")
    import traceback
    traceback.print_exc()

print("\nDone.")

#python /home/pintokf/Projects/Microbium/Mouses/MIPMLP_scripts/run_mipmlp_preprocessing.py