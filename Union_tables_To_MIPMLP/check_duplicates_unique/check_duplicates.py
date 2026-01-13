import pandas as pd

# נתיב לקובץ המאוחד
file_path = "Union_tables_To_MIPMLP/check_duplicates_unique/final_merged_table.csv"

print(f"Checking for duplicates in: {file_path}")

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("Error: File not found. Make sure the merge script finished successfully.")
    exit(1)

# בדיקת שורות כפולות (לפי ה-ID בעמודה הראשונה)
# אנו מניחים שהעמודה הראשונה היא ה-FeatureID
id_column = df.columns[0]
print(f"Using column '{id_column}' as the unique identifier.")

# ספירת החזרות
duplicates = df[df.duplicated(subset=[id_column], keep=False)]
num_duplicates = len(duplicates)

print("-" * 30)
if num_duplicates == 0:
    print("✅ SUCCESS: No duplicates found.")
    print(f"There are {len(df)} unique bacteria (ASVs) in the table.")
else:
    print(f"❌ WARNING: Found {num_duplicates} duplicate rows!")
    print("Here are the first few duplicates:")
    print(duplicates.head())
print("-" * 30)
