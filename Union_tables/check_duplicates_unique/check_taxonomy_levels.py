import pandas as pd

# נתיב לקובץ המאוחד שיצרנו בשלב הקודם
file_path = "Union_tables/final_merged_table.csv"

print("Loading table...")
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("Error: Could not find the file. Make sure you ran the merge script first.")
    exit(1)

# חיפוש עמודת הטקסונומיה (בדרך כלל נקראת 'Taxon' או 'Taxon_y' אם היה כפילות)
# נחפש עמודה שמכילה את המילה Taxon
tax_col = [col for col in df.columns if 'Taxon' in col]
if not tax_col:
    # אם לא מצאנו לפי שם, נניח שזו העמודה האחרונה (בדרך כלל זה המצב בקובץ merged)
    target_col = df.columns[-1]
    print(f"Warning: 'Taxon' column not found by name. Using last column: '{target_col}'")
else:
    target_col = tax_col[0]

print(f"Analyzing column: {target_col}\n")

# === פונקציות ספירה ===

# בדיקה האם יש Genus (מחפשים 'g__' שיש אחריו טקסט, ולא סתם ריק)
def has_genus(text):
    if pd.isna(text): return False
    if "g__" not in text: return False
    # מפצלים לפי g__ ולוקחים את מה שאחריו
    content = text.split("g__")[-1].split(";")[0] # לוקחים עד ה-; הבא
    return len(content) > 0 and content != "uncultured" # מוודאים שיש תוכן

# בדיקה האם יש Species (מחפשים 's__' שיש אחריו טקסט)
def has_species(text):
    if pd.isna(text): return False
    if "s__" not in text: return False
    content = text.split("s__")[-1].split(";")[0]
    return len(content) > 0 and content != "uncultured_bacterium"

# === חישוב ===
total_rows = len(df)
count_g = df[target_col].apply(has_genus).sum()
count_s = df[target_col].apply(has_species).sum()

print("-" * 40)
print(f"Total ASVs (Bacteria types): {total_rows}")
print("-" * 40)
print(f"Genus Level (g) Identified:   {count_g}  ({(count_g/total_rows)*100:.1f}%)")
print(f"Species Level (s) Identified: {count_s}  ({(count_s/total_rows)*100:.1f}%)")
print("-" * 40)

if count_s < total_rows * 0.1:
    print("\nNOTE: Low species detection (<10%).")
    print("This is common with short reads (like 150bp) or aggressive trimming.")