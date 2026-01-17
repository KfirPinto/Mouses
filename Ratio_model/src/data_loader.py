import pandas as pd
import re

def clean_col_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))

def load_and_prep_data(censored_path, uncensored_path, age_filter=None):
    print(f"Loading data from:\n {censored_path}\n {uncensored_path}")
    
    censored = pd.read_csv(censored_path, index_col=0)
    uncensored = pd.read_csv(uncensored_path, index_col=0)

    # Clean columns
    censored.columns = [clean_col_name(c) for c in censored.columns]
    uncensored.columns = [clean_col_name(c) for c in uncensored.columns]

    # Filter by Age if defined in yaml
    if age_filter is not None:
        print(f"Filtering for Age: {age_filter}")
        # מוצא את עמודת הגיל באופן דינמי (מכילה 'Age')
        c_age_col = [c for c in censored.columns if "Age" in c][0]
        u_age_col = [c for c in uncensored.columns if "Age" in c][0]
        
        censored = censored[censored[c_age_col] == age_filter]
        uncensored = uncensored[uncensored[u_age_col] == age_filter]

    # Extract Cage IDs
    censored["Cage"] = [str(i).split("-")[0] for i in censored.index]
    uncensored["Cage"] = [str(i).split("-")[0] for i in uncensored.index]

    # Remove rows with NaN in target column (critical for feature selection to work)
    target_col = [c for c in uncensored.columns if "diff" in c.lower()][0]
    uncensored_before = len(uncensored)
    uncensored = uncensored.dropna(subset=[target_col])
    if len(uncensored) < uncensored_before:
        print(f"Removed {uncensored_before - len(uncensored)} samples with NaN in target column")

    print(f"Loaded: {len(censored)} censored, {len(uncensored)} uncensored samples.")
    return censored, uncensored