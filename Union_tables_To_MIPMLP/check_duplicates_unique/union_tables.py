import pandas as pd
import os

# === הגדרת נתיבים ===
# וודא שהשמות תואמים בדיוק למה שיש לך בתיקייה
otu_path = "mouses_data/clean_fastq/exports/otu.csv"

# שים לב: לפעמים הטקסונומיה נמצאת בתוך תת-תיקייה או בשם tax.tsv
# שנה את השורה הזו אם הקובץ שלך נקרא אחרת
tax_path = "mouses_data/clean_fastq/exports/tax.tsv/taxonomy.csv" 

output_dir = "Union_tables"
output_filename = "final_merged_table.csv"

# === בדיקת תקינות נתיבים ===
if not os.path.exists(output_dir):
    print(f"Creating directory: {output_dir}")
    os.makedirs(output_dir)

print("Loading OTU table...")
try:
    # קריאת קובץ ה-OTU (מדלגים על שורה ראשונה אם היא הערה של biom)
    otu_df = pd.read_csv(otu_path, header=0)
    
    # אחידות: משנים את שם העמודה הראשונה (ה-ID) לשם קבוע כדי למנוע בלבול
    otu_df.rename(columns={otu_df.columns[0]: 'FeatureID'}, inplace=True)
    print(f"OTU Table loaded: {otu_df.shape[0]} rows, {otu_df.shape[1]} columns")

except Exception as e:
    print(f"Error loading OTU file: {e}")
    exit(1)

print("Loading Taxonomy table...")
try:
    # קריאת קובץ הטקסונומיה
    tax_df = pd.read_csv(tax_path, header=0)
    
    # אחידות: משנים גם כאן את העמודה הראשונה ל-FeatureID
    tax_df.rename(columns={tax_df.columns[0]: 'FeatureID'}, inplace=True)
    print(f"Taxonomy Table loaded: {tax_df.shape[0]} rows")

except Exception as e:
    print(f"Error loading Taxonomy file: {e}")
    print("Tip: Check if the file is named 'taxonomy.tsv' or lies inside a folder.")
    exit(1)

# === ביצוע האיחוד (Merge) ===
print("Merging tables based on FeatureID...")

# אנחנו עושים 'inner' join כדי לשמור רק שורות שקיימות בשני הקבצים
# (בדרך כלל הם זהים ב-100%)
merged_df = pd.merge(otu_df, tax_df, on='FeatureID', how='inner')

# === שמירה ===
output_path = os.path.join(output_dir, output_filename)
merged_df.to_csv(output_path, index=False)

print("-" * 30)
print(f"DONE! Merged file saved to:\n{output_path}")
print(f"Final shape: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
print("-" * 30)