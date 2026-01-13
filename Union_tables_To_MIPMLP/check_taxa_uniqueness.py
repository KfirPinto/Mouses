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

# פונקציות חילוץ (משמשות רק כדי לדעת אם השורה היא Unassigned או לא)
def has_genus(text):
    if pd.isna(text) or "g__" not in text: return False
    val = text.split("g__")[-1].split(";")[0]
    return val != "" and val != "Unassigned"

def has_species(text):
    if pd.isna(text) or "s__" not in text: return False
    val = text.split("s__")[-1].split(";")[0]
    return val != "" and val != "Unassigned"

# === בדיקת Genus ===
# סינון: לוקחים רק שורות שיש להן סיווג Genus חוקי
df_genus_assigned = df[df[tax_col].apply(has_genus)]

# ספירה: "יוניק" נחשב רק אם כל הנתיב הטקסונומי שונה
unique_genera_paths = df_genus_assigned[tax_col].nunique()
total_assigned_g = len(df_genus_assigned)

print(f"--- Genus Level (g) ---")
print(f"Total rows with assigned Genus: {total_assigned_g}")
print(f"Unique Genus Paths (Full Taxonomy): {unique_genera_paths}")

if total_assigned_g > unique_genera_paths:
    print(f"NOTE: There are {total_assigned_g - unique_genera_paths} duplicated rows (exact same path).")
    print("Most common Genus Paths (Top 3 duplicates):")
    # ספירת החזרות של הנתיב המלא
    print(df_genus_assigned[tax_col].value_counts().head(3))
else:
    print("No duplicates found at Genus level (based on full path).")

print("\n" + "-"*50 + "\n")

# === בדיקת Species ===
# סינון: לוקחים רק שורות שיש להן סיווג Species חוקי
df_species_assigned = df[df[tax_col].apply(has_species)]

# ספירה: "יוניק" נחשב רק אם כל הנתיב הטקסונומי שונה
unique_species_paths = df_species_assigned[tax_col].nunique()
total_assigned_s = len(df_species_assigned)

print(f"--- Species Level (s) ---")
print(f"Total rows with assigned Species: {total_assigned_s}")
print(f"Unique Species Paths (Full Taxonomy): {unique_species_paths}")

if total_assigned_s > unique_species_paths:
    print(f"NOTE: There are {total_assigned_s - unique_species_paths} duplicated rows (exact same path).")
    print("Most common Species Paths (Top 3 duplicates):")
    # ספירת החזרות של הנתיב המלא
    print(df_species_assigned[tax_col].value_counts().head(3))
else:
    print("No duplicates found at Species level (based on full path).")