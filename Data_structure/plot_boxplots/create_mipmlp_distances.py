import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# --- 1. הגדרת נתיבים ---
# משתמשים בקובץ המוכן מ-MIPMLP במקום בקובץ הגולמי!
mipmlp_file = "/home/pintokf/Projects/Microbium/Mouses/MIPMLP_scripts/whole_metadata_whole_samples/processed_subpca_level6.csv"
meta_file = "/home/pintokf/Projects/Microbium/Mouses/Data_structure/to_catagorial_metadata/metadata_categorical.tsv"
output_plot = "/home/pintokf/Projects/Microbium/Mouses/Data_structure/MIPMLP_Distance_Boxplots.png"

# --- 2. קריאת הנתונים מ-MIPMLP ---
# הקובץ מכיל עמודה בשם 'ID' שבה יושבים שמות הדגימות, לכן נגדיר אותה כאינדקס.
df_feat = pd.read_csv(mipmlp_file, index_col='ID')

# קריאת המטא-דאטה (בשביל לדעת מי זה איזה עכבר ומאיזה גיל)
df_meta = pd.read_csv(meta_file, sep='\t', index_col=0)
if df_meta.index[0] == '#q2:types':
    df_meta = df_meta.drop('#q2:types')

# מוודאים שאנחנו עובדים רק על דגימות שקיימות בשני הקבצים
common = df_feat.index.intersection(df_meta.index)
df_feat = df_feat.loc[common]
df_meta = df_meta.loc[common]

print(f"Working on {len(common)} samples...")

# --- 3. חישוב מרחק אוקלידי ---
# מכיוון ש-MIPMLP כבר עשה את הטרנספורמציות, אנחנו יכולים לחשב מרחק ישירות!
dist_array = pdist(df_feat.values, metric='euclidean')
dist_matrix = pd.DataFrame(squareform(dist_array), index=df_feat.index, columns=df_feat.index)

# --- 4. סיווג ל-3 הקבוצות שהמנחה ביקש ---
pairs = []
samples = dist_matrix.columns

print("Calculating pairwise distances and classifying groups (this takes a few seconds)...")
for i in range(len(samples)):
    for j in range(i+1, len(samples)): # משולש עליון למניעת כפילויות
        s1, s2 = samples[i], samples[j]
        d = dist_matrix.iloc[i, j]
        
        # הוצאת פרטי העכברים והזמן (אם העמודה שלך נקראת 'number' ולא 'mice_name', שנה כאן)
        mouse1, time1 = df_meta.loc[s1, 'mice_name'], df_meta.loc[s1, 'AgeMonth']
        mouse2, time2 = df_meta.loc[s2, 'mice_name'], df_meta.loc[s2, 'AgeMonth']
        
        # סיווג ל-3 ה-Boxplots
        if time1 == time2 and mouse1 != mouse2:
            category = "Same Time\nDifferent Mice"
        elif mouse1 == mouse2 and time1 != time2:
            category = "Same Mouse\nDifferent Times"
        elif mouse1 != mouse2 and time1 != time2:
            category = "Different Mice\nDifferent Times"
        else:
            category = "Exclude" # אותו עכבר באותו זמן (כפילות טכנית אם יש)
            
        if category != "Exclude":
            pairs.append({'Distance': d, 'Category': category})

df_pairs = pd.DataFrame(pairs)

# --- 5. ציור הגרף (Boxplots) ---
plt.figure(figsize=(10, 7))

# הגדרת סדר העמודות בגרף
order = ["Same Time\nDifferent Mice", "Same Mouse\nDifferent Times", "Different Mice\nDifferent Times"]

# ציור הקופסאות בצבעים שונים
sns.boxplot(x='Category', y='Distance', data=df_pairs, order=order, 
            palette=['#66c2a5', '#fc8d62', '#8da0cb'], width=0.6, fliersize=2)

plt.title('Pairwise Euclidean Distances Between Microbiome Samples\n(Based on MIPMLP Log-Normalized Features)', 
          fontsize=15, fontweight='bold', pad=15)
plt.ylabel('Euclidean Distance', fontsize=13, fontweight='bold')
plt.xlabel('')
plt.xticks(fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# הוספת כיתוב עם כמויות הזוגות לכל קופסה (בונוס נחמד למאמרים)
counts = df_pairs['Category'].value_counts()
for i, cat in enumerate(order):
    plt.text(i, df_pairs['Distance'].min() - (df_pairs['Distance'].max()*0.05), 
             f"n={counts[cat]} pairs", ha='center', fontsize=10, color='grey')

plt.tight_layout()
plt.savefig(output_plot, dpi=300)
print(f"\nהגרף נוצר בהצלחה! תמצא אותו כאן:\n{output_plot}")