import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

# --- 1. הגדרת נתיבים ---
mipmlp_file = "/home/pintokf/Projects/Microbium/Mouses/MIPMLP_scripts/whole_metadata_whole_samples/processed_subpca_level6.csv"
meta_file = "/home/pintokf/Projects/Microbium/Mouses/Data_structure/to_catagorial_metadata/metadata_categorical.tsv"
output_plot = "/home/pintokf/Projects/Microbium/Mouses/Data_structure/mi_cluster/Microbe_MI_Heatmap_Clean.png"

# --- 2. קריאת הנתונים ---
df_feat = pd.read_csv(mipmlp_file, index_col='ID')
df_meta = pd.read_csv(meta_file, sep='\t', index_col=0)
if df_meta.index[0] == '#q2:types': 
    df_meta = df_meta.drop('#q2:types')

df = df_feat.join(df_meta[['mice_name', 'AgeMonth']]).dropna(subset=['mice_name', 'AgeMonth'])
microbes = df_feat.columns.tolist()

# --- 3. הכנה לחישוב MI ---
base_month = 'Month_02'
all_months = sorted(df['AgeMonth'].unique(), key=lambda x: int(x.split('_')[1]))
target_months = [m for m in all_months if m != base_month]

mi_results = pd.DataFrame(index=microbes, columns=target_months)

print("Calculating Mutual Information...")

for microbe in microbes:
    pivot = df.pivot_table(index='mice_name', columns='AgeMonth', values=microbe, aggfunc='mean')
    
    for tm in target_months:
        if base_month in pivot.columns and tm in pivot.columns:
            subset = pivot[[base_month, tm]].dropna()
            
            if len(subset) >= 10: 
                X = subset[[base_month]] 
                y = subset[tm]
                mi = mutual_info_regression(X, y, random_state=42)[0]
                mi_results.loc[microbe, tm] = mi

mi_results = mi_results.dropna(how='all').astype(float).fillna(0)

# --- 4. סינון וניקוי (טופ 15 בלבד) ---
def clean_taxon_name(name):
    parts = [p.strip() for p in name.split(';') if not p.strip().endswith('__')]
    return parts[-1] if parts else name

def make_unique(labels):
    seen = {}
    new_labels = []
    for label in labels:
        if label not in seen:
            seen[label] = 0
            new_labels.append(label)
        else:
            seen[label] += 1
            new_labels.append(f"{label} ({seen[label]})")
    return new_labels

# ניקח את 15 החיידקים שהיה להם את פיק ה-MI הגבוה ביותר ונסדר אותם בסדר יורד
top_microbes = mi_results.max(axis=1).sort_values(ascending=False).head(15).index
mi_results = mi_results.loc[top_microbes]

cleaned_names = [clean_taxon_name(m) for m in mi_results.index]
mi_results.index = make_unique(cleaned_names)
mi_results.columns = [c.replace('Month_', '') + ' Mo' for c in mi_results.columns]

# --- 5. יצירת מפת חום נקייה ---
print("Generating Clean Heatmap...")
plt.figure(figsize=(10, 8)) # גודל קומפקטי ומתאים ל-15 שורות

# משתמשים ב-heatmap רגיל (ללא קלאסטרינג)
sns.heatmap(mi_results, 
            cmap='magma_r', 
            cbar_kws={'label': 'Mutual Information (MI)'},
            linewidths=0.5,       # מוסיף קווי מתאר למשבצות שיהיה נעים לעין
            linecolor='lightgrey',
            annot=True,           # כותב את המספרים בתוך המשבצות!
            fmt=".2f",            # 2 ספרות אחרי הנקודה
            annot_kws={"size": 10})

plt.title('Top 15 Microbes: Personal Signature Over Time (vs Month 02)\n(Higher MI = Stronger Personal Baseline)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(rotation=0, fontsize=11)
plt.ylabel('') 

plt.tight_layout()
plt.savefig(output_plot, dpi=300)
print(f"הגרף הנקי נוצר בהצלחה! תמצא אותו ב:\n{output_plot}")