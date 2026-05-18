import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import math

# --- 1. טעינת נתונים (מ-MIPMLP) ---
mipmlp_file = "/home/pintokf/Projects/Microbium/Mouses/MIPMLP_scripts/whole_metadata_whole_samples/processed_subpca_level6.csv"
meta_file = "/home/pintokf/Projects/Microbium/Mouses/Data_structure/to_catagorial_metadata/metadata_categorical.tsv"
output_plot = "/home/pintokf/Projects/Microbium/Mouses/Data_structure/alpha_diversity_month_2/Beta_Personal_Path_ALL.png"

df_feat = pd.read_csv(mipmlp_file, index_col='ID')
df_meta = pd.read_csv(meta_file, sep='\t', index_col=0)
if df_meta.index[0] == '#q2:types': 
    df_meta = df_meta.drop('#q2:types')

# שמירת דגימות משותפות בלבד
common = df_feat.index.intersection(df_meta.index)
df_feat = df_feat.loc[common]
df_meta = df_meta.loc[common]

# --- 2. פונקציה לייצור מטריצת מרחקים פר חודש ---
def get_dist_matrix(month):
    # מציאת הדגימות של החודש הספציפי
    samples = df_meta[df_meta['AgeMonth'] == month].index
    
    # טיפול בכפילויות (אם נדגם אותו עכבר פעמיים באותו חודש, ניקח רק דגימה אחת כדי שאפשר יהיה להשוות)
    meta_subset = df_meta.loc[samples].drop_duplicates(subset=['mice_name'])
    valid_samples = meta_subset.index
    mice_names = meta_subset['mice_name'].values
    
    if len(valid_samples) < 10: # אם אין לפחות 10 עכברים, לא נחשב
        return None
        
    data = df_feat.loc[valid_samples]
    # חישוב מרחק אוקלידי בין כל העכברים בחודש הזה
    dist = squareform(pdist(data.values, metric='euclidean'))
    return pd.DataFrame(dist, index=mice_names, columns=mice_names)

# --- 3. השוואת כל החודשים לחודש 2 ---
base_month = 'Month_02'
base_dist = get_dist_matrix(base_month)

all_months = sorted(df_meta['AgeMonth'].unique(), key=lambda x: int(x.split('_')[1]))
target_months = [m for m in all_months if m != base_month]

valid_targets = []
target_dist_dict = {}

for tm in target_months:
    tm_dist = get_dist_matrix(tm)
    if tm_dist is not None:
        # בודקים כמה עכברים משותפים יש בין חודש 2 לחודש המטרה
        common_mice = base_dist.index.intersection(tm_dist.index)
        if len(common_mice) >= 10:
            valid_targets.append(tm)
            target_dist_dict[tm] = tm_dist

# --- 4. ציור הגרפים המלאים ---
cols = 4 
rows = math.ceil(len(valid_targets) / cols)
plt.figure(figsize=(4 * cols, 4 * rows))

for i, tm in enumerate(valid_targets, 1):
    tm_dist = target_dist_dict[tm]
    common_mice = base_dist.index.intersection(tm_dist.index)
    
    # חיתוך המטריצות כך שיכילו רק את העכברים המשותפים ובאותו סדר בדיוק!
    m1 = base_dist.loc[common_mice, common_mice]
    m2 = tm_dist.loc[common_mice, common_mice]
    
    # הוצאת המרחקים (המשולש העליון של המטריצה, בלי האלכסון)
    vec1 = m1.values[np.triu_indices(len(common_mice), k=1)]
    vec2 = m2.values[np.triu_indices(len(common_mice), k=1)]
    
    # בדיקת קורלציה
    corr, pval = spearmanr(vec1, vec2)
    
    plt.subplot(rows, cols, i)
    sns.regplot(x=vec1, y=vec2, scatter_kws={'alpha':0.5, 'color':'purple'}, line_kws={'color':'black'})
    
    # נצבע את הרקע בכחול בהיר אם הקורלציה מובהקת
    if pval < 0.05:
        plt.gca().set_facecolor('#f0f8ff')
        
    tm_name = tm.replace('Month_', '') + " Mo"
    plt.title(f'Mo 02 vs {tm_name}\nr={corr:.2f}, p={pval:.3f}', fontweight='bold')
    plt.xlabel('Pairwise Distances (Month 02)')
    plt.ylabel(f'Pairwise Distances ({tm_name})')

plt.suptitle("Does Initial Distance Predict Future Distance? (Beta Diversity)", fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"הגרף של השוואת המרחקים נוצר בהצלחה!\nנשמר ב: {output_plot}")