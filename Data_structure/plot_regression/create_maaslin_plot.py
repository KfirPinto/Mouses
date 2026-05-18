import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# --- נתיבים ---
input_file = "/home/pintokf/Projects/Microbium/Mouses/Data_structure/Maaslin2_Results/significant_results.tsv"
output_plot = "/home/pintokf/Projects/Microbium/Mouses/Data_structure/Maaslin2_Results/MaAsLin_Significant_Taxa_Python.png"

# קריאת הנתונים
df = pd.read_csv(input_file, sep='\t')

# --- 1. סינון זבל (מטא דאטה ושמות לא מזוהים) ---
bad_features = ["date", "number", "AgeWeeks", "Plate", "AgeMonth", "AgeNumeric", "death_age_month"]
df_clean = df[~df['feature'].isin(bad_features)].copy()

# סינון עמודות שמתחילות ב-X עם מספרים (כמו X.1, X.2)
df_clean = df_clean[~df_clean['feature'].str.match(r'^X(\.[0-9]+)?$')]

# --- 2. ניקוי שמות חיידקים ארוכים ---
def clean_name(name):
    name = re.sub(r'k__Bacteria\.p__.*\.f__', 'f__', name)
    name = re.sub(r'k__Bacteria\.p__.*\.o__', 'o__', name)
    name = re.sub(r'k__Bacteria\.p__.*\.c__', 'c__', name)
    name = name.replace('.__', '')
    name = name.replace('k__Bacteria', 'Unclassified Bacteria')
    return name

df_clean['feature'] = df_clean['feature'].apply(clean_name)

# --- 3. בחירת 15 החיידקים הכי מובהקים (qval הכי נמוך) ---
top_taxa = df_clean.sort_values('qval').head(15).copy()

# הוספת כיווניות ומיון לפי המקדם (כדי שזה ייראה טוב בגרף)
top_taxa['Direction'] = top_taxa['coef'].apply(lambda x: 'Increased with Age' if x > 0 else 'Decreased with Age')
top_taxa = top_taxa.sort_values('coef', ascending=False)

# --- 4. ציור הגרף ---
plt.figure(figsize=(10, 7))
sns.set_theme(style="whitegrid")

# קביעת הצבעים לפי הכיוון
colors = ['#d73027' if d == 'Increased with Age' else '#4575b4' for d in top_taxa['Direction']]

# יצירת הפלוט האופקי
ax = sns.barplot(x='coef', y='feature', data=top_taxa, palette=colors, edgecolor='black', linewidth=0.5)

# כותרות ועיצוב
plt.title('Significant Microbial Changes with Age\nMaAsLin2 Multivariable Linear Association', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Model Coefficient (Effect Size)', fontsize=12, fontweight='bold')
plt.ylabel('')
plt.yticks(fontsize=11, fontstyle='italic')

# יצירת מקרא מותאם אישית
import matplotlib.patches as mpatches
inc_patch = mpatches.Patch(color='#d73027', label='Increased with Age')
dec_patch = mpatches.Patch(color='#4575b4', label='Decreased with Age')
plt.legend(handles=[inc_patch, dec_patch], loc='lower right', fontsize=11)

# שמירה
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
print(f"\nהגרף נוצר ונשמר בהצלחה בנתיב:\n{output_plot}")