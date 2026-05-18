import os

# הגדרת נתיבים
input_path = '/home/pintokf/Projects/Microbium/Mouses/mouse_data_new/metadata_all_samples_new.tsv'
out_dir = '/home/pintokf/Projects/Microbium/Mouses/Data_structure/to_catagorial_metadata'
output_path = f'{out_dir}/metadata_categorical.tsv'

# יצירת התיקייה החדשה במידה והיא לא קיימת
os.makedirs(out_dir, exist_ok=True)

with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
    lines = infile.readlines()
    
    # מציאת המיקום של עמודת הגיל
    header = lines[0].strip('\n').split('\t')
    try:
        age_idx = header.index('AgeMonth')
    except ValueError:
        print("Error: Column 'AgeMonth' not found in the header!")
        exit()
        
    outfile.write(lines[0])
    
    # מעבר על שאר השורות ושינוי הערכים
    for line in lines[1:]:
        # דילוג על שורות ריקות לחלוטין
        if not line.strip():
            continue
            
        cols = line.strip('\n').split('\t')
        
        # מוודאים שהשורה ארוכה מספיק כדי להכיל את עמודת הגיל
        if len(cols) > age_idx:
            if cols[0] == '#q2:types':
                cols[age_idx] = 'categorical'
            else:
                try:
                    val = int(float(cols[age_idx]))
                    cols[age_idx] = f"Month_{val:02d}"
                except ValueError:
                    pass # במידה והתא ריק או שכבר מכיל טקסט
        
        outfile.write('\t'.join(cols) + '\n')

print(f"Success! Categorical metadata saved to:\n{output_path}")