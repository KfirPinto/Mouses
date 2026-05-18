import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. קריאת הקובץ מ-QIIME 2 ---
# (שנה את שם הקובץ או הנתיב אם צריך)
df = pd.read_csv("/home/pintokf/Projects/Microbium/Mouses/Data_structure/plot_shannon/kruskal-wallis-pairwise-AgeMonth.csv")

# --- 2. ניקוי שמות הקבוצות ---
# חותך את החלק של ה-(n=58) כדי שהטבלה תיראה נקייה
df['Group 1'] = df['Group 1'].str.split(' \(').str[0]
df['Group 2'] = df['Group 2'].str.split(' \(').str[0]

# מציאת כל החודשים הייחודיים וסידורם בסדר מספרי עולה
groups = pd.concat([df['Group 1'], df['Group 2']]).unique()
groups = sorted(groups, key=lambda x: int(x.split('_')[1]))

n = len(groups)

# --- 3. יצירת מטריצות ריקות ---
h_matrix = pd.DataFrame(np.zeros((n, n)), columns=groups, index=groups)
p_matrix = pd.DataFrame(np.ones((n, n)), columns=groups, index=groups)

# מילוי המטריצות בנתונים מתוך הקובץ
for _, row in df.iterrows():
    g1, g2 = row['Group 1'], row['Group 2']
    h, p = row['H'], row['p-value']
    
    # מכיוון שההשוואה היא זוגית, ממלאים באופן סימטרי
    h_matrix.loc[g1, g2] = h
    h_matrix.loc[g2, g1] = h
    p_matrix.loc[g1, g2] = p
    p_matrix.loc[g2, g1] = p

# --- 4. הכנת הכיתוב בתוך המשבצות (Annotations) ---
annotations = pd.DataFrame(np.full((n, n), ''), columns=groups, index=groups)

for i in range(n):
    for j in range(n):
        if i == j: # האלכסון (עצמו מול עצמו)
            annotations.iloc[i, j] = '-'
            h_matrix.iloc[i, j] = np.nan # מונע צביעה של האלכסון
        elif i < j: # משולש עליון: p-values
            pval = p_matrix.iloc[i, j]
            if pval < 0.001:
                annotations.iloc[i, j] = 'p<0.001'
            else:
                annotations.iloc[i, j] = f'p={pval:.3f}'
        else: # משולש תחתון: H scores (Kruskal-Wallis)
            annotations.iloc[i, j] = f'H={h_matrix.iloc[i, j]:.1f}'

# --- 5. יצירת הפלוט (המטריצה היפה) ---
plt.figure(figsize=(12, 10))

# ציור מפת החום (Heatmap)
heatmap = sns.heatmap(h_matrix, annot=annotations, fmt='', cmap='coolwarm', 
                      cbar_kws={'label': 'H-Score (Effect Size)'}, 
                      annot_kws={"size": 11, "weight": "bold"})

# הוספת מסגרות שחורות עבות למשבצות שיצאו מובהקות (p < 0.05)
for i in range(n):
    for j in range(n):
        if i != j and p_matrix.iloc[i, j] < 0.05:
            heatmap.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=3))

# עיצוב אחרון
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12, rotation=0)
plt.title('Shannon Diversity Pairwise Comparisons\n(Upper: p-value | Lower: H-score)', 
          fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()

# שמירת התמונה
output_plot = "Shannon_Matrix_Plot.png"
plt.savefig(output_plot, dpi=300)
print(f"המטריצה נוצרה בהצלחה ונשמרה בשם: {output_plot}")