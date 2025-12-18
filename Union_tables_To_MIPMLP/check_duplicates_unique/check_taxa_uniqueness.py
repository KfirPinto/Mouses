import pandas as pd

# נתיב לקובץ
file_path = "Union_tables/final_merged_table.csv"

try:
    df = pd.read_csv(file_path)
except:
    print("Error loading file.")
    exit(1)

# מציאת עמודת הטקסונומיה
tax_col = [col for col in df.columns if 'Taxon' in col][0]

# פונקציות חילוץ נקיות
def get_genus(text):
    if pd.isna(text) or "g__" not in text: return "Unassigned"
    val = text.split("g__")[-1].split(";")[0]
    return val if val else "Unassigned"

def get_species(text):
    if pd.isna(text) or "s__" not in text: return "Unassigned"
    val = text.split("s__")[-1].split(";")[0]
    return val if val else "Unassigned"

# יצירת עמודות זמניות לבדיקה
df['Genus_Only'] = df[tax_col].apply(get_genus)
df['Species_Only'] = df[tax_col].apply(get_species)

# === בדיקת Genus ===
unique_genera = df['Genus_Only'].nunique()
total_assigned_g = len(df[df['Genus_Only'] != "Unassigned"])
print(f"--- Genus Level (g) ---")
print(f"Total rows with Genus: {total_assigned_g}")
print(f"Unique Genus names:    {unique_genera}")
if total_assigned_g > unique_genera:
    print(f"NOTE: There are {total_assigned_g - unique_genera} rows that share a Genus name with others.")
    print("Most common Genera:")
    print(df[df['Genus_Only'] != "Unassigned"]['Genus_Only'].value_counts().head(3))

print("\n" + "-"*30 + "\n")

# === בדיקת Species ===
unique_species = df['Species_Only'].nunique()
total_assigned_s = len(df[df['Species_Only'] != "Unassigned"])
print(f"--- Species Level (s) ---")
print(f"Total rows with Species: {total_assigned_s}")
print(f"Unique Species names:    {unique_species}")
if total_assigned_s > unique_species:
    print(f"NOTE: There are {total_assigned_s - unique_species} rows that share a Species name with others.")
    print("Most common Species:")
    print(df[df['Species_Only'] != "Unassigned"]['Species_Only'].value_counts().head(3))

