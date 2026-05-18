import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import PLSCanonical
from scipy.stats import pearsonr

# --- 1. נתיבים ---
metab_file = "/home/pintokf/Projects/Microbium/Mouses/preprocess_metabolits/preprocessed_metabolites_normalized_z_score.csv"
microbe_file = "/home/pintokf/Projects/Microbium/Mouses/MIPMLP_scripts/whole_metadata_whole_samples/processed_subpca_level6.csv"
meta_file = "/home/pintokf/Projects/Microbium/Mouses/Data_structure/to_catagorial_metadata/metadata_categorical.tsv"
output_plot = "/home/pintokf/Projects/Microbium/Mouses/Data_structure/metabolites/Metabolites_Microbes_CCA.png"

# --- 2. קריאת מטא-דאטה ---
df_meta = pd.read_csv(meta_file, sep='\t', index_col=0)
if df_meta.index[0] == '#q2:types': 
    df_meta = df_meta.drop('#q2:types')

# --- 3. קריאת וארגון נתוני מטבוליטים ---
df_met = pd.read_csv(metab_file, index_col=0)

if df_met.shape[0] > df_met.shape[1]:
    df_met = df_met.T

print(f"Total samples in metabolites file (before fix): {len(df_met)}")

# --- התיקון שלנו: תיקון השמות של המטבוליטים ---
def fix_metabolite_id(old_id):
    try:
        # בודק אם יש קו תחתון וקו אמצעי (כמו 18-1_5-20)
        if '_' in old_id and '-' in old_id.split('_')[-1]:
            parts = old_id.split('_') # ['18-1', '5-20']
            date_parts = parts[1].split('-') # ['5', '20']
            month = date_parts[0].zfill(2) # מוסיף אפס מוביל: '5' הופך ל-'05'
            year = date_parts[1] # '20'
            return f"{parts[0]}_{month}_{year}"
    except:
        pass
    return old_id

# החלת התיקון על כל שמות הדגימות בטבלת המטבוליטים
df_met.index = [fix_metabolite_id(idx) for idx in df_met.index]

# חיבור למטא-דאטה עכשיו כשהשמות תואמים!
df_met_meta = df_met.join(df_meta[['mice_name', 'AgeMonth']]).dropna(subset=['mice_name', 'AgeMonth'])
print(f"Samples after joining with metadata: {len(df_met_meta)}")

if len(df_met_meta) == 0:
    print("\nERROR: Still no matching IDs. Let's check the data further.")
    exit()

met_m2 = df_met_meta[df_met_meta['AgeMonth'] == 'Month_02']
print(f"Samples in Month 02 for metabolites: {len(met_m2)}")

# ממוצע לעכבר כדי למנוע כפילויות
met_m2_mice = met_m2.drop(columns=['AgeMonth', 'mice_name']).groupby(met_m2['mice_name']).mean()
print(f"Unique mice in Month 02 for metabolites: {len(met_m2_mice)}\n")

# לוקחים 100 מטבוליטים עם שונות מקסימלית (Sparsity)
top_metabs = met_m2_mice.var().sort_values(ascending=False).head(100).index
met_m2_mice = met_m2_mice[top_metabs]

# --- 4. קריאת וארגון נתוני מיקרוביום ---
df_mic = pd.read_csv(microbe_file, index_col='ID')
df_mic_meta = df_mic.join(df_meta[['mice_name', 'AgeMonth']]).dropna(subset=['mice_name', 'AgeMonth'])

# --- 5. הרצת CCA לכל חודש מטרה ---
all_months = sorted(df_meta['AgeMonth'].dropna().unique(), key=lambda x: int(x.split('_')[1]))
target_months = [m for m in all_months if m != 'Month_02']

results = []
print("Running Canonical Correlation Analysis (CCA)...")

for tm in target_months:
    mic_tm = df_mic_meta[df_mic_meta['AgeMonth'] == tm]
    if len(mic_tm) == 0:
        continue
        
    mic_tm_mice = mic_tm.drop(columns=['AgeMonth', 'mice_name']).groupby(mic_tm['mice_name']).mean()
    
    # חיתוך עכברים משותפים בין מטבוליטים בחודש 2 למיקרוביום בחודש המטרה
    common_mice = met_m2_mice.index.intersection(mic_tm_mice.index)
    print(f"Target {tm}: Found {len(common_mice)} common mice.")
    
    if len(common_mice) >= 5: 
        X = met_m2_mice.loc[common_mice]
        Y = mic_tm_mice.loc[common_mice]
        
        cca = PLSCanonical(n_components=1)
        X_c, Y_c = cca.fit_transform(X, Y)
        
        if np.std(X_c[:, 0]) > 0 and np.std(Y_c[:, 0]) > 0:
            corr, pval = pearsonr(X_c[:, 0], Y_c[:, 0])
            
            results.append({
                'Target_Month': tm.replace('Month_', '') + ' Mo',
                'Correlation': corr,
                'P_value': pval,
                'N_mice': len(common_mice)
            })

df_results = pd.DataFrame(results)

if df_results.empty:
    print("\nERROR: Still no results to plot. The number of common mice is too low across all target months.")
else:
    # --- 6. ציור הגרף ---
    plt.figure(figsize=(10, 6))

    sns.barplot(x='Target_Month', y='Correlation', data=df_results, palette='viridis', edgecolor='black')

    for i, row in df_results.iterrows():
        if row['P_value'] < 0.05:
            plt.text(i, row['Correlation'] + 0.02, '*', ha='center', fontsize=20, color='red', fontweight='bold')
        plt.text(i, max(0.05, row['Correlation']/2), f"n={row['N_mice']}", ha='center', color='white', fontweight='bold')

    plt.title('Predictive Power of Month 02 Metabolites on Future Microbiome\n(Canonical Correlation Analysis)', 
              fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Canonical Correlation (R)', fontsize=12, fontweight='bold')
    plt.xlabel('Target Microbiome Month', fontsize=12, fontweight='bold')
    plt.ylim(0, 1.1)

    plt.axhline(0.7, color='red', linestyle='--', alpha=0.5)
    plt.text(len(df_results)-0.5, 0.72, 'High Correlation', color='red', va='bottom', ha='right')

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"\nגרף ה-CCA נוצר בהצלחה! תוצאות נשמרו ב:\n{output_plot}")