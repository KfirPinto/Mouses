import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# --- 1. נתיבים לקבצים ---
shannon_file = "/home/pintokf/Projects/Microbium/Mouses/Data_structure/shannon_exported/alpha-diversity.tsv"
meta_file = "/home/pintokf/Projects/Microbium/Mouses/Data_structure/to_catagorial_metadata/metadata_categorical.tsv"
output_plot = "/home/pintokf/Projects/Microbium/Mouses/Data_structure/plot_shannon/Shannon_U_Matrix.png"

# --- 2. קריאת הנתונים ---
df_shannon = pd.read_csv(shannon_file, sep='\t', index_col=0)
shannon_col = df_shannon.columns[0]

df_meta = pd.read_csv(meta_file, sep='\t', index_col=0)
if df_meta.index[0] == '#q2:types':
    df_meta = df_meta.drop('#q2:types')

df_merged = df_meta.join(df_shannon).dropna(subset=[shannon_col, 'AgeMonth'])

# --- 3. הכנת הקבוצות ---
unique_months = df_merged['AgeMonth'].unique()
months_sorted = sorted(unique_months, key=lambda x: int(x.split('_')[1]))

n = len(months_sorted)

u_matrix = pd.DataFrame(np.zeros((n, n)), columns=months_sorted, index=months_sorted)
p_matrix = pd.DataFrame(np.ones((n, n)), columns=months_sorted, index=months_sorted)

# --- 4. חישוב Mann-Whitney U לכל זוג ---
for i in range(n):
    for j in range(n):
        if i >= j:
            continue
            
        group1 = months_sorted[i]
        group2 = months_sorted[j]
        
        vals1 = df_merged[df_merged['AgeMonth'] == group1][shannon_col]
        vals2 = df_merged[df_merged['AgeMonth'] == group2][shannon_col]
        
        stat, pval = mannwhitneyu(vals1, vals2, alternative='two-sided')
        
        median_diff = vals1.median() - vals2.median()
        directional_u = stat if median_diff >= 0 else -stat
        
        u_matrix.loc[group1, group2] = directional_u
        u_matrix.loc[group2, group1] = directional_u
        p_matrix.loc[group1, group2] = pval
        p_matrix.loc[group2, group1] = pval

# --- 5. יצירת האנוטציות (הטקסט בתוך המשבצות) ---
annotations = pd.DataFrame(np.full((n, n), ''), columns=months_sorted, index=months_sorted)
display_names = [str(int(m.split('_')[1])) + " Mo" for m in months_sorted]

for i in range(n):
    for j in range(n):
        if i == j:
            annotations.iloc[i, j] = '-'
            u_matrix.iloc[i, j] = np.nan
        elif i < j: 
            pval = p_matrix.iloc[i, j]
            if pval < 0.001:
                annotations.iloc[i, j] = 'p<0.001'
            else:
                annotations.iloc[i, j] = f'p={pval:.3f}'
        else: 
            annotations.iloc[i, j] = f'U={u_matrix.iloc[i, j]:.0f}'

# --- 6. ציור מפת החום ---
plt.figure(figsize=(12, 10))

# השינוי כאן: חזרנו ל-coolwarm הבוהק והוספנו center=0
heatmap = sns.heatmap(u_matrix, annot=annotations, fmt='', cmap='coolwarm', center=0, 
                      cbar_kws={'label': 'Signed U-Score (Effect Size & Direction)'}, 
                      xticklabels=display_names, yticklabels=display_names,
                      annot_kws={"size": 11, "weight": "bold"})

for i in range(n):
    for j in range(n):
        if i != j and p_matrix.iloc[i, j] < 0.05:
            heatmap.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=3))

plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12, rotation=0)
plt.title('Shannon Diversity Pairwise Comparisons\n(Upper: p-value | Lower: Signed Mann-Whitney U-score)', 
          fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(output_plot, dpi=300)
print(f"המטריצה הכיוונית המעודכנת (בצבעים בוהקים) נוצרה בהצלחה: {output_plot}")