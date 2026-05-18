import pandas as pd

# הגדרת הנתיבים לקבצים
metadata_path = '/home/pintokf/Projects/Microbium/Mouses/mouses_2_data/metadata_all_samples_new.csv'
preprocess_path = '/home/pintokf/Projects/Microbium/Mouses/Union_tables_To_MIPMLP/for_preprocess.csv'

# קריאת הקבצים
# משתמשים ב-dtype=str כדי לוודא שכל ה-IDs נקראים כטקסט
df_metadata = pd.read_csv(metadata_path, dtype=str)
df_preprocess = pd.read_csv(preprocess_path, dtype=str)

# שליפת העמודות הרלוונטיות והמרתן ל-Sets (קבוצות)
metadata_ids = set(df_metadata['#SampleID'])
preprocess_ids = set(df_preprocess['ID'])

# מציאת ה-ID שקיים במטא-דאטה אבל חסר בקובץ העיבוד המקדים
missing_ids = metadata_ids - preprocess_ids

# הדפסת התוצאות
if missing_ids:
    print(f"\n✓ Found {len(missing_ids)} missing ID(s) in for_preprocess.csv:")
    for missing_id in missing_ids:
        print(f"  - {missing_id}")
else:
    print("\n✓ No missing IDs found. All IDs from metadata are present in for_preprocess.csv.")