import pandas as pd

# נתיב לקובץ
file_path = "Union_tables_To_MIPMLP/check_duplicates_unique/final_merged_table.csv"

try:
    df = pd.read_csv(file_path)
except:
    print("Error: File not found.")
    exit(1)

# מציאת עמודת הטקסונומיה
tax_col = [col for col in df.columns if 'Taxon' in col][0]

# --- פונקציות לחילוץ הנתיב המלא ---

def get_full_genus_path(text):
    # אם אין זיהוי סוג, מחזירים כלום
    if pd.isna(text) or "g__" not in text: return None
    
    # בדיקה האם ה-g__ ריק (למשל "g__;")
    genus_part = text.split("g__")[-1].split(";")[0]
    if len(genus_part) < 1 or genus_part == "uncultured": return None

    # החזרת הנתיב המלא עד סוף ה-Genus
    # (מוצאים איפה נגמר ה-Genus וחותכים שם)
    end_index = text.find("g__") + len("g__") + len(genus_part)
    return text[:end_index]

def get_full_species_path(text):
    # אם אין זיהוי מין, מחזירים כלום
    if pd.isna(text) or "s__" not in text: return None
    
    # בדיקה האם ה-s__ ריק
    species_part = text.split("s__")[-1].split(";")[0]
    if len(species_part) < 1 or "uncultured" in species_part: return None

    # החזרת הנתיב המלא עד סוף ה-Species
    end_index = text.find("s__") + len("s__") + len(species_part)
    return text[:end_index]

# --- ביצוע הניתוח ---

# יצירת עמודות של נתיבים מלאים
df['Full_Genus_Path'] = df[tax_col].apply(get_full_genus_path)
df['Full_Species_Path'] = df[tax_col].apply(get_full_species_path)

print(f"Analyzing Full Taxonomy Paths (Lineages)...")
print("-" * 40)

# 1. ניתוח Genus
total_g_rows = df['Full_Genus_Path'].notna().sum()
unique_g_paths = df['Full_Genus_Path'].nunique()

print(f"GENUS LEVEL (Full Path):")
print(f"Total Rows Identified:   {total_g_rows}")
print(f"Unique Lineages found:   {unique_g_paths}")

if total_g_rows > unique_g_paths:
    print(f"-> Conclusion: NOT UNIQUE. {total_g_rows - unique_g_paths} rows share the exact same biological lineage.")
else:
    print(f"-> Conclusion: ALL UNIQUE.")

print("-" * 40)

# 2. ניתוח Species
total_s_rows = df['Full_Species_Path'].notna().sum()
unique_s_paths = df['Full_Species_Path'].nunique()

print(f"SPECIES LEVEL (Full Path):")
print(f"Total Rows Identified:   {total_s_rows}")
print(f"Unique Lineages found:   {unique_s_paths}")

if total_s_rows > unique_s_paths:
    print(f"-> Conclusion: NOT UNIQUE. {total_s_rows - unique_s_paths} rows share the exact same biological lineage.")
    print("Example of duplication (Full Path):")
    print(df['Full_Species_Path'].value_counts().head(3))
else:
    print(f"-> Conclusion: ALL UNIQUE.")
print("-" * 40)