# 1. Imports for suppression MUST come first
import warnings
import pandas as pd
import matplotlib.pyplot as plt

# 2. AGGRESSIVE WARNING SUPPRESSION
# These lines must run before importing LBL/MLE_augmentor
warnings.filterwarnings("ignore") 
pd.options.mode.chained_assignment = None  # This kills the specific SettingWithCopyWarning

# 3. Now import the library (it will inherit the settings above)
from sklearn.model_selection import GroupShuffleSplit
from LBL import LBL

if __name__ == '__main__':
    # load censored and uncensored data
    censored = pd.read_csv("/home/pintokf/Projects/Microbium/Mouses/Ratio_model/Preprocces_for_ratio_model/data_level7_censored.csv", index_col=0)
    censored["Cage"] = [i.split("-")[0] for i in censored.index]
    censored = censored[censored["AgeMonths"] == 2]
    uncensored = pd.read_csv("/home/pintokf/Projects/Microbium/Mouses/Ratio_model/Preprocces_for_ratio_model/data_level7_uncensored.csv", index_col=0)
    uncensored["Cage"] = [i.split("-")[0] for i in uncensored.index]
    uncensored = uncensored[uncensored["AgeMonths"] == 2]


    # list_of_categories = ['Gander', 'HDM', 'AtopicDermatitis', 'Asthma']

    # create the LBL class
    lbl = LBL("diff", "MiceName", "AgeMonths", num_of_bact=35, feature_selection=35, with_microbiome=True,
              augmented_censored=False, gamma=0.0, only_microbiome=True, alpha=0.001)

    # train test split uncensored

    gss = GroupShuffleSplit(n_splits=2, train_size=.7)
    gss.get_n_splits()
    for train_idx, test_idx in gss.split(uncensored, groups=uncensored["Cage"]):
        train_uncensored = uncensored.iloc[train_idx]
        test_uncensored = uncensored.iloc[test_idx]

    lbl.fit(train_uncensored, censored)

    predictions = lbl.predict(test_uncensored)
    print(predictions)

    print(lbl.score(test_uncensored, test_uncensored["diff"]))