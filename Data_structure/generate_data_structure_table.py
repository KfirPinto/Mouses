#!/usr/bin/env python3
"""
Generate data structure summary table from metadata
- Age in months = ceil(age_weeks / 4)
- Death status: from death_age_month column (number = dead, na = alive)
- Check which samples have metabolites data
"""

import os
import pandas as pd
import numpy as np
import csv

def main():
    # Read metadata
    input_txt = '/home/pintokf/Projects/Microbium/Mouses/mouses_2_data/metadata_all_samples_new.txt'
    converted_csv = '/home/pintokf/Projects/Microbium/Mouses/mouses_2_data/metadata_all_samples_new.csv'

    # Read TXT and save as CSV
    with open(input_txt, 'r', encoding='utf-8') as txt_file, \
         open(converted_csv, 'w', newline='', encoding='utf-8') as csv_file:
        reader = csv.reader(txt_file, delimiter='\t')
        writer = csv.writer(csv_file)
        for row in reader:
            writer.writerow(row)

    df = pd.read_csv(converted_csv, sep=',')
    df = df.dropna(how='all').copy()
    
    # Convert columns to appropriate types
    df['AgeMonth'] = pd.to_numeric(df['AgeMonth'], errors='coerce')
    df['metabolomics'] = df['metabolomics'].str.lower().str.strip()
    df['death'] = df['death'].str.lower().str.strip()
    
    print("\n" + "="*80)
    print("DATA STRUCTURE SUMMARY")
    print("="*80)
    print(f"Total samples: {len(df)}")
    
    # Count unique mice by mice_name
    unique_mice = df['mice_name'].nunique()
    
    # Count alive and dead mice
    alive_mice = df[df['death'] == 'no']['mice_name'].nunique()
    dead_mice = df[df['death'] == 'yes']['mice_name'].nunique()
    
    print(f"\nTotal unique mice: {unique_mice}")
    print(f"  Alive: {alive_mice}")
    print(f"  Dead: {dead_mice}")
    
    # Group by AgeMonth
    age_months = sorted(df['AgeMonth'].dropna().unique())
    
    print("\n" + "="*80)
    print("BREAKDOWN BY AGE MONTH")
    print("="*80)
    
    # Prepare data for CSV
    summary_data = []
    
    for age in age_months:
        age = int(age)
        age_data = df[df['AgeMonth'] == age]
        
        # Count alive (death == 'no')
        alive = age_data[age_data['death'] == 'no']
        alive_count = len(alive)
        alive_with_meta = len(alive[alive['metabolomics'] == 'yes'])
        
        # Count dead (death == 'yes')
        dead = age_data[age_data['death'] == 'yes']
        dead_count = len(dead)
        dead_with_meta = len(dead[dead['metabolomics'] == 'yes'])
        
        # Calculate death age mean ± std
        death_age_str = "NA ± NA"
        death_age_mean = None
        death_age_std = None
        if len(dead) > 0:
            death_ages = pd.to_numeric(dead['death_age_month'], errors='coerce').dropna()
            if len(death_ages) > 0:
                death_age_mean = death_ages.mean()
                death_age_std = death_ages.std()
                death_age_str = f"{death_age_mean:.2f} ± {death_age_std:.2f}"
        
        print(f"\nAge {age} months:")
        print(f"  Alive: {alive_count} samples ({alive_with_meta} with metabolomics)")
        print(f"  Dead: {dead_count} samples ({dead_with_meta} with metabolomics)")
        print(f"  Death age: {death_age_str} months")
        
        # Add to summary data
        summary_data.append({
            'AgeMonth': age,
            'Alive_N': alive_count,
            'Alive_Metabolomics': alive_with_meta,
            'Dead_N': dead_count,
            'Dead_Metabolomics': dead_with_meta,
            'Death_Age_Mean': death_age_mean,
            'Death_Age_Std': death_age_std,
            'Death_Age': death_age_str
        })
    
    # Save to CSV
    summary_df = pd.DataFrame(summary_data)
    output_path = '/home/pintokf/Projects/Microbium/Mouses/Data_structure/data_structure_summary_all_new.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"\n✓ Summary saved to '{output_path}'")


if __name__ == '__main__':
    main()
