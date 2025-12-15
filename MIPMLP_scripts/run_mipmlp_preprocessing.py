import pandas as pd
import MIPMLP
import os

# === הגדרות ===
input_path = "/home/pintokf/Projects/Microbium/Mouses/Union_tables/for_preprocess.csv"
output_dir = "/home/pintokf/Projects/Microbium/Mouses/MIPMLP_scripts"

print(f"Loading data from: {input_path}")
try:
    # 1. טעינת הקובץ (העמודה הראשונה היא האינדקס)
    df = pd.read_csv(input_path, index_col=0)
    
    # 2. התיקון הקריטי: הופכים את האינדקס לעמודה רגילה בשם ID
    #df.index.name = 'ID'
    df.reset_index(inplace=True)
    
    print(f"Data loaded & Index reset. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:5]} ...") # בדיקה שה-ID קיים
    
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# ==========================================
# ריצה 1: רמה 7 (Species) - SubPCA
# ==========================================
print("\n--- Processing Level 7 (Species) ---")
try:
    processed_L7 = MIPMLP.preprocess(
        df,
        taxonomy_level=7,
        taxnomy_group='mean',
        normalization='log',
        drop_tax_prefix=True,
        plot=False,
        epsilon=0.00001,
        z_scoring='No',
        pca=(0, 'PCA'),
        rare_bacteria_threshold=0.01
    )
    
    if isinstance(processed_L7, tuple):
        processed_L7 = processed_L7[0]

    out_path = f"{output_dir}/processed_subpca_level7_mean.csv"
    processed_L7.to_csv(out_path, index=False)
    print(f"✅ Saved Level 7 to: {out_path}")

except Exception as e:
    print(f"❌ Error in Level 7: {e}")
    import traceback
    traceback.print_exc()

# ==========================================
# ריצה 2: רמה 6 (Genus) - SubPCA
# ==========================================
print("\n--- Processing Level 6 (Genus) ---")
try:
    processed_L6 = MIPMLP.preprocess(
        df,
        taxonomy_level=6,
        taxnomy_group='mean',
        normalization='log',
        drop_tax_prefix=True,
        plot=False,
        epsilon=0.00001,
        z_scoring='No',
        pca=(0, 'PCA'),
        rare_bacteria_threshold=0.01
    )

    if isinstance(processed_L6, tuple):
        processed_L6 = processed_L6[0]

    out_path = f"{output_dir}/processed_subpca_level6_mean.csv"
    processed_L6.to_csv(out_path, index=False)
    print(f"✅ Saved Level 6 to: {out_path}")

except Exception as e:
    print(f"❌ Error in Level 6: {e}")
    import traceback
    traceback.print_exc()

print("\nDone.")

#python /home/pintokf/Projects/Microbium/Mouses/MIPMLP_scripts/run_mipmlp_preprocessing.py