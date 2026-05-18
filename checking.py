import pandas as pd

# נתיב לקובץ שלך
file_path = '/home/pintokf/Projects/Microbium/Mouses/mouse_data_new/metadata_all_samples.tsv'

# קריאת הקובץ (מפריד טאב)
df = pd.read_csv(file_path, sep='\t')

# 1. בדיקה אם הן זהות לחלוטין
are_identical = df['death'].equals(df['death_s'])

print(f"Are 'death' and 'death_s' identical? {are_identical}")

if not are_identical:
    print("\nFound differences:")
    # יצירת טבלה שמראה רק את השורות שבהן יש הבדל
    differences = df[df['death'] != df['death_s']][['#SampleID', 'death', 'death_s']]
    print(differences)
else:
    print("\nYou can safely remove one of them.")