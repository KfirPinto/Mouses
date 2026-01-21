import pandas as pd
import os

# רשימת הקבצים המלאה
file_paths = [
    "/home/pintokf/Projects/Microbium/Mouses/results/Metabolites/Ratio_age2/hyper_summary.csv",
    "/home/pintokf/Projects/Microbium/Mouses/results/Metabolites/Ratio_age4/hyper_summary.csv",
    "/home/pintokf/Projects/Microbium/Mouses/results/Metabolites/Ratio_unfiltered/hyper_summary.csv",
    "/home/pintokf/Projects/Microbium/Mouses/results/Whole_data_level_6/Locate_age2_inference/hyper_summary.csv",
    "/home/pintokf/Projects/Microbium/Mouses/results/Whole_data_level_6/Locate_age4_inference/hyper_summary.csv",
    "/home/pintokf/Projects/Microbium/Mouses/results/Whole_data_level_6/Locate_age6_inference/hyper_summary.csv",
    "/home/pintokf/Projects/Microbium/Mouses/results/Whole_data_level_6/Locate_unfiltered_inference/hyper_summary.csv",
    "/home/pintokf/Projects/Microbium/Mouses/results/Whole_data_level_7/Locate_age2_inference/hyper_summary.csv",
    "/home/pintokf/Projects/Microbium/Mouses/results/Whole_data_level_7/Locate_age4_inference/hyper_summary.csv",
    "/home/pintokf/Projects/Microbium/Mouses/results/Whole_data_level_7/Locate_age6_inference/hyper_summary.csv",
    "/home/pintokf/Projects/Microbium/Mouses/results/Whole_data_level_7/Locate_unfiltered_inference/hyper_summary.csv",
    "/home/pintokf/Projects/Microbium/Mouses/results/Whole_data_level_6/Microbium_age2/hyper_summary.csv",
    "/home/pintokf/Projects/Microbium/Mouses/results/Whole_data_level_6/Microbium_age4/hyper_summary.csv",
    "/home/pintokf/Projects/Microbium/Mouses/results/Whole_data_level_6/Microbium_age6/hyper_summary.csv",
    "/home/pintokf/Projects/Microbium/Mouses/results/Whole_data_level_6/Microbium_unfiltered/hyper_summary.csv",
    "/home/pintokf/Projects/Microbium/Mouses/results/Whole_data_level_7/Microbium_age2/hyper_summary.csv",
    "/home/pintokf/Projects/Microbium/Mouses/results/Whole_data_level_7/Microbium_age4/hyper_summary.csv",
    "/home/pintokf/Projects/Microbium/Mouses/results/Whole_data_level_7/Microbium_age6/hyper_summary.csv",
    "/home/pintokf/Projects/Microbium/Mouses/results/Whole_data_level_7/Microbium_unfiltered/hyper_summary.csv"
]

dataframes = []

for path in file_paths:
    try:
        # קריאת הקובץ
        df = pd.read_csv(path)
        
        # --- לוגיקה עבור taxonomy_level ---
        if 'level_6' in path:
            df['taxonomy_level'] = 6
        elif 'level_7' in path:
            df['taxonomy_level'] = 7
        else:
            df['taxonomy_level'] = 0
            
        # --- לוגיקה עבור filter ---
        if 'age2' in path:
            df['filter'] = 2
        elif 'age4' in path:
            df['filter'] = 4
        elif 'age6' in path:
            df['filter'] = 6
        elif 'unfiltered' in path:
            df['filter'] = 0
        else:
            df['filter'] = -1 

        # --- לוגיקה עבור העמודות הבינאריות ועמודת number ---
        # מאתחלים ב-0
        df['Metabolites'] = 0
        df['Locate'] = 0
        df['Microbium'] = 0
        df['number'] = 0  # ברירת מחדל
        
        if 'Metabolites' in path:
            df['Metabolites'] = 1
            df['number'] = 1889
        elif 'Locate' in path:
            df['Locate'] = 1
            df['number'] = 10
        elif 'Microbium' in path:
            df['Microbium'] = 1
            df['number'] = 35
            
        dataframes.append(df)
        print(f"Processed: {path}")
        
    except FileNotFoundError:
        print(f"Error: File not found - {path}")

# איחוד כל הדאטה-פריימים
if dataframes:
    final_df = pd.concat(dataframes, ignore_index=True)
    
    # הגדרת נתיב השמירה
    output_dir = "/home/pintokf/Projects/Microbium/Mouses/Union_results"
    
    # יצירת התיקייה אם היא לא קיימת
    os.makedirs(output_dir, exist_ok=True)
    
    # שמירת הקובץ
    output_path = os.path.join(output_dir, "united_hyper_summary.csv")
    final_df.to_csv(output_path, index=False)
    
    print("-" * 30)
    print(f"Successfully created union file at:\n{output_path}")
    print(f"Total rows: {len(final_df)}")
else:
    print("No files were processed.")