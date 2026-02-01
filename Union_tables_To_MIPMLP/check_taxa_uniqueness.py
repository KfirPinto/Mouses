import pandas as pd

# נתיב לקובץ (מותאם לקובץ שהעלית או לנתיב שלך)
file_path = "Union_tables_To_MIPMLP/check_duplicates_unique/final_merged_table.csv" # או הנתיב המקורי שלך: "Union_tables_To_MIPMLP/check_duplicates_unique/final_merged_table.csv"

try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# מציאת עמודת הטקסונומיה באופן אוטומטי
try:
    tax_col = [col for col in df.columns if 'Taxon' in col or 'taxonomy' in col.lower()][0]
    print(f"Working on taxonomy column: {tax_col}\n")
except IndexError:
    print("Could not find a column with 'Taxon' in its name.")
    exit(1)

# פונקציות חילוץ - מזהה אם יש g__ או s__ בטקסט (גם אם ריק)
def has_genus(text):
    if pd.isna(text): return False
    return "g__" in text

def has_species(text):
    if pd.isna(text): return False
    return "s__" in text

# === בדיקת כל השורות (ALL) ===
total_rows = len(df)
unique_all_paths = df[tax_col].nunique()

print(f"=== ALL TAXONOMY PATHS ===")
print(f"Total rows: {total_rows}")
print(f"Unique Full Taxonomy Paths: {unique_all_paths}")

if total_rows > unique_all_paths:
    print(f"Duplicates: {total_rows - unique_all_paths} rows share the exact same full taxonomy path.")
    print("\nMost common paths (Top 5):")
    print(df[tax_col].value_counts().head(5))
else:
    print("All rows are unique (no duplicates).")

print("\n" + "-"*50 + "\n")

# === בדיקת Genus (כולל g__ ריק) ===
# כל שורה שיש בה g__ (גם אם ריק) - ספירה לפי הנתיב המלא
df_genus = df[df[tax_col].apply(has_genus)]

unique_genera_paths = df_genus[tax_col].nunique()
total_g = len(df_genus)

print(f"=== GENUS LEVEL (including empty g__) ===")
print(f"Total rows with g__: {total_g}")
print(f"Unique Paths: {unique_genera_paths}")

if total_g > unique_genera_paths:
    print(f"Duplicates: {total_g - unique_genera_paths} rows share the exact same path.")
    print("\nMost common Genus-level paths (Top 5):")
    print(df_genus[tax_col].value_counts().head(5))
else:
    print("All genus-level rows are unique.")

print("\n" + "-"*50 + "\n")

# === בדיקת Species (כולל s__ ריק) ===
# כל שורה שיש בה s__ (גם אם ריק) - ספירה לפי הנתיב המלא
df_species = df[df[tax_col].apply(has_species)]

unique_species_paths = df_species[tax_col].nunique()
total_s = len(df_species)

print(f"=== SPECIES LEVEL (including empty s__) ===")
print(f"Total rows with s__: {total_s}")
print(f"Unique Paths: {unique_species_paths}")

if total_s > unique_species_paths:
    print(f"Duplicates: {total_s - unique_species_paths} rows share the exact same path.")
    print("\nMost common Species-level paths (Top 5):")
    print(df_species[tax_col].value_counts().head(5))
else:
    print("All species-level rows are unique.")