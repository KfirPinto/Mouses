import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import math

# --- 1. טעינת נתונים ---
shannon_file = "/home/pintokf/Projects/Microbium/Mouses/Data_structure/shannon_exported/alpha-diversity.tsv"
meta_file = "/home/pintokf/Projects/Microbium/Mouses/Data_structure/to_catagorial_metadata/metadata_categorical.tsv"

df_shannon = pd.read_csv(shannon_file, sep='\t', index_col=0)
df_meta = pd.read_csv(meta_file, sep='\t', index_col=0)
if df_meta.index[0] == '#q2:types': 
    df_meta = df_meta.drop('#q2:types')

df = df_meta.join(df_shannon).dropna(subset=['shannon_entropy', 'AgeMonth', 'mice_name'])

# ארגון הנתונים לפי עכברים (ממוצע לכפילויות)
path_df = df.pivot_table(index='mice_name', columns='AgeMonth', values='shannon_entropy', aggfunc='mean')

base_month = 'Month_02'
all_other_months = [col for col in path_df.columns if col != base_month]

# סינון חודשים שיש להם לפחות 5 עכברים משותפים עם חודש 2
valid_targets = []
for t in all_other_months:
    if len(path_df[[base_month, t]].dropna()) >= 5:
        valid_targets.append(t)

# --- ציור הגרף המלא ---
n_plots = len(valid_targets)
cols = 4 # 4 גרפים בשורה
rows = math.ceil(n_plots / cols)

plt.figure(figsize=(4 * cols, 4 * rows))

for i, target in enumerate(valid_targets, 1):
    subset = path_df[[base_month, target]].dropna()
    corr, pval = spearmanr(subset[base_month], subset[target])
    
    plt.subplot(rows, cols, i)
    sns.regplot(x=base_month, y=target, data=subset, color='darkgreen', scatter_kws={'alpha':0.7})
    
    # צביעת הרקע באדום בהיר אם התוצאה מובהקת, כדי שיבלוט
    if pval < 0.05:
        plt.gca().set_facecolor('#fff0f0')
        
    # ניקוי השמות (Month_04 -> 4)
    target_name = target.replace('Month_', '') + " Months"
    
    plt.title(f'Mo 02 vs {target_name}\nr={corr:.2f}, p={pval:.3f}', fontweight='bold')
    plt.xlabel('Shannon at Month 02')
    plt.ylabel(f'Shannon at {target_name}')

plt.suptitle("Does Month 2 Determine Future Alpha Diversity?", fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
output_file = "/home/pintokf/Projects/Microbium/Mouses/Data_structure/alpha_diversity_month_2/Alpha_Personal_Path_ALL.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"הגרף המלא נוצר בהצלחה!\nנשמר ב: {output_file}")