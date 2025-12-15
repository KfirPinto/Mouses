import pandas as pd
import numpy as np
import os
from ratio_t2e import RatioT2E
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr

# === הגדרות נתיבים ===
base_dir = "/home/pintokf/Projects/Microbium/Mouses/MIPMLP_scripts"
metadata_path = "/home/pintokf/Projects/Microbium/Mouses/mouses_2_data/metadata.txt" # שים לב: תוודא ששמרת את הקובץ בשם הזה
microbiome_L6_path = f"{base_dir}/processed_subpca_level6.csv"
microbiome_L7_path = f"{base_dir}/processed_subpca_level7.csv"
output_results = f"/home/pintokf/Projects/Microbium/Mouses/Ratio_model/ratio_results.csv"

# === 1. פונקציה לטעינת וסידור המטה-דאטה ===
def load_and_clean_metadata(path):
    print(f"Loading metadata from: {path}")
    # המטה-דאטה נראה מופרד בטאבים או רווחים
    df = pd.read_csv(path, sep='\t') 
    
    # ניקוי רווחים בשמות העמודות
    df.columns = [c.strip() for c in df.columns]
    
    # חילוץ ID
    if '#SampleID' in df.columns:
        df.rename(columns={'#SampleID': 'ID'}, inplace=True)
    df.set_index('ID', inplace=True)
    
    # יצירת עמודת Event (E)
    # yes = 1 (Uncensored), no = 0 (Censored)
    df['E'] = df['Death'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    
    # יצירת עמודת Time (T)
    # המרה למספרים, טיפול ב-na
    df['DeathAgeMonths'] = pd.to_numeric(df['DeathAgeMonths'], errors='coerce')
    
    # === טיפול בחסרים (Censored) ===
    # מי שמת (E=1) מקבל את זמן התמותה שלו
    # מי ששרד (E=0) מקבל את הזמן המקסימלי שנצפה בניסוי (או שאפשר להגדיר מספר קבוע ידנית)
    max_observed_time = df['DeathAgeMonths'].max()
    print(f"Max observed death age: {max_observed_time} months")
    
    # מילוי זמנים חסרים: אם אין זמן תמותה, נניח שהם שרדו עד סוף הניסוי (ניקח את המקסימום + 1)
    # הערה: אם יש לך גיל סיום מדויק לכל עכבר ששרד, עדיף להשתמש בו!
    censored_time_fill = max_observed_time + 1 
    df['T'] = df['DeathAgeMonths'].fillna(censored_time_fill)
    
    return df[['E', 'T']]

# === 2. פונקציה להרצת Ratio וחישוב מדדים ===
def run_ratio_analysis(microbiome_path, meta_df, level_name):
    print(f"\n--- Running Ratio on {level_name} ---")
    
    # טעינת המיקרוביום
    try:
        micro_df = pd.read_csv(microbiome_path, index_col='ID')
    except Exception as e:
        print(f"❌ Error loading microbiome file: {e}")
        return None

    # איחוד הטבלאות (לפי ID)
    # לוקחים רק דגימות שקיימות גם במטה-דאטה וגם במיקרוביום
    merged_df = micro_df.join(meta_df, how='inner')
    print(f"Samples matched: {len(merged_df)} (out of {len(micro_df)} in microbiome)")
    
    if len(merged_df) < 5:
        print("❌ Not enough samples to train!")
        return None

    # הכנת הנתונים למודל
    X = merged_df.drop(columns=['E', 'T']).values
    T = merged_df['T'].values
    E = merged_df['E'].values
    
    # === הרצת Ratio ===
    # פרמטרים בסיסיים, אפשר לשחק איתם
    model = RatioT2E(hidden_dims=[32, 16], batch_size=8, epochs=100, lr=0.005, verbose=False)
    
    # אימון (על כל הדאטה כרגע, כי ביקשת הערכה כללית. במחקר אמיתי עושים Train/Test split)
    model.fit(X, T, E)
    
    # חיזוי
    predicted_times = model.predict(X)
    
    # === הערכת ביצועים ===
    # 1. C-Index (מודד דיוק סדרתי על כולם)
    c_index = concordance_index(T, predicted_times, E)
    
    # 2. קורלציה (רק על ה-Uncensored, מי שמת בפועל)
    uncensored_idx = (E == 1)
    if uncensored_idx.sum() > 2:
        real_T_uncensored = T[uncensored_idx]
        pred_T_uncensored = predicted_times[uncensored_idx]
        
        corr, p_val = pearsonr(real_T_uncensored, pred_T_uncensored)
    else:
        corr, p_val = 0, 1
        print("⚠️ Not enough uncensored samples for correlation.")

    print(f"✅ Results for {level_name}:")
    print(f"   C-Index: {c_index:.4f}")
    print(f"   Pearson Correlation (Uncensored): {corr:.4f} (p={p_val:.4f})")
    
    return {
        "Level": level_name,
        "C-Index": c_index,
        "Correlation": corr,
        "P-Value": p_val,
        "Num_Samples": len(merged_df)
    }

# === Main Pipeline ===
if __name__ == "__main__":
    # 1. הכנת המטה-דאטה
    # (מניחים שקובץ המטה-דאטה קיים בנתיב שהגדרנו)
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata file not found at: {metadata_path}")
        print("Please upload/save the metadata file first.")
        exit(1)
        
    meta_df = load_and_clean_metadata(metadata_path)
    
    results = []
    
    # 2. הרצה על רמה 6 (Genus)
    res_L6 = run_ratio_analysis(microbiome_L6_path, meta_df, "Level 6 (Genus)")
    if res_L6: results.append(res_L6)
    
    # 3. הרצה על רמה 7 (Species)
    res_L7 = run_ratio_analysis(microbiome_L7_path, meta_df, "Level 7 (Species)")
    if res_L7: results.append(res_L7)
    
    # 4. סיכום
    print("\n=== Final Summary ===")
    summary_df = pd.DataFrame(results)
    print(summary_df)
        
    # שמירת התוצאות
    summary_df.to_csv(output_results, index=False)
    print(f"\nSaved summary to: {output_results}")
