import pandas as pd
import numpy as np

# נתיבים לקבצים
input_csv = '/home/pintokf/Projects/Microbium/Mouses/Data_structure/exported_all_samples/level-2.csv'
output_tsv = '/home/pintokf/Projects/Microbium/Mouses/Data_structure/fb_ratio.tsv'

# קריאת הנתונים
df = pd.read_csv(input_csv, index_col=0)

# חיפוש העמודות הרלוונטיות (שמות ארוכים שכוללים את שם החיידק)
firm_col = [col for col in df.columns if 'Firmicutes' in col][0]
bact_col = [col for col in df.columns if 'Bacteroidetes' in col][0]

# חישוב היחס (מוסיפים מספר פצפון כדי למנוע קריסה אם יש חלוקה ב-0)
fb_ratio = df[firm_col] / (df[bact_col] + 1e-9)

# שמירה בפורמט ש-QIIME 2 מזהה כמדד (עם #SampleID)
out_df = pd.DataFrame({'FB_ratio': fb_ratio})
out_df.index.name = '#SampleID'

out_df.to_csv(output_tsv, sep='\t')
print(f"Success! F/B ratio saved to: {output_tsv}")