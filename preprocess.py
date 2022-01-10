import pandas as pd 
import json
from sklearn.preprocessing import LabelEncoder

def main():
    
    # Load settings
    with open("settings.json", "r") as inFile:
        settings = json.load(inFile)
        
    # Load TCR data 
    tcr_act = pd.read_csv(settings["file"]["TCR_activated"])
    tcr_neg = pd.read_csv(settings["file"]["TCR_negative"])
    
    # Concatenate negative and positive activations
    colnames = ["CDR3ab", "activated_by"]
    tcr_df = pd.concat([tcr_act[colnames], tcr_neg[colnames]])
    
    # Tokenize labels, and create individual lables for HA69 and NP136
    le = LabelEncoder().fit(tcr_df.activated_by)
    tcr_df["num_label"] = le.transform(tcr_df.activated_by)
    tcr_df["activatedby_HA"] = tcr_df.num_label.apply(lambda x: 1 if x in [0] else 0)
    tcr_df["activatedby_HCRT"] = tcr_df.num_label.apply(lambda x: 1 if x in [3] else 0)
    tcr_df["activatedby_NP"] = tcr_df.num_label.apply(lambda x: 1 if x in [5] else 0)
    tcr_df["negative"] = tcr_df.num_label.apply(lambda x: 1 if x == 6 else 0)
    tcr_df["activated_any"] = tcr_df.num_label.apply(lambda x: 1 if x != 6 else 0)
    
    
    # Write dataframe
    tcr_df.to_csv(settings["file"]["TCR_data"], header=True, index=False)


if __name__ == "__main__":
    main()