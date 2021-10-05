import numpy as np
import pandas as pd 
import json
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers import pre_tokenizers
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD
from tokenizers.pre_tokenizers import ByteLevel

def get_token_train_data(settings:dict) -> list():
    """get_token_train_data [Reads data, splits train and test, writes them, and 
            writes a file with the training data for the tokenizer]

    Args:
        settings (dict): [Settingsfile]
    """
    
    # Read data 
    data_act = pd.read_csv(settings["file"]["TCR_activated"])
    data_neg = pd.read_csv(settings["file"]["TCR_negative"])

    # Merge 
    cols = ["CDR3ab", "activated_by"]
    tcr_df = pd.concat([data_act[cols], data_neg[cols]])
    
    # Split train test and merge
    X_train, X_test = train_test_split(tcr_df, test_size=settings["param"]["test_split"])
    
    # Tokenize labels , and create individual lables for HA69 and NP136
    le = LabelEncoder().fit(tcr_df.activated_by)
    tcr_df["num_label"] = le.transform(tcr_df.activated_by)
    tcr_df["activatedby_HA"] = tcr_df.num_label.apply(lambda x: 1 if x in [0,1,2] else 0)
    tcr_df["activatedby_HCRT"] = tcr_df.num_label.apply(lambda x: 1 if x in [1,3,4] else 0)
    tcr_df["activatedby_NP"] = tcr_df.num_label.apply(lambda x: 1 if x in [2,4,5] else 0)
    tcr_df["activated_any"] = tcr_df.num_label.apply(lambda x: 1 if x != 6 else 0)
    
    # Write dataframe
    tcr_df.to_csv(settings["file"]["TCR_data"], header=True, index=False)
    
    # Write file to train tokenizer 
    with open(settings["file"]["tokenizer_data"],"w") as outFile:
        for cdr in X_train.CDR3ab:
            outFile.write(cdr + "\n")
                
    
def tokenization_pipeline(settings:dict) -> None:
    """tokenization_pipeline [Reads the training data and trains the tokenizer]

    Args:
        settings (dict): [Settings file]
    """
    
    # Create normalizer and pre-tokenizer
    normalizer = normalizers.Sequence([Lowercase(), NFD()])
    pre_tokenizer = pre_tokenizers.Sequence([ByteLevel()])
    
    # Create tokenizer
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    
    # Train on data 
    tokenizer.train(files=settings["file"]["tokenizer_data"], min_frequency = 2)
    tokenizer.save_model(settings["dir"]["Resources"])

def main():
    
    # Load settings 
    with open("settings.json","r") as inFile:
        settings = json.load(inFile)
        
    # Set random seeds
    seed_nr = 1964
    torch.manual_seed(seed_nr)
    np.random.seed(seed_nr)
        
    # Create file with tokenizer training data
    get_token_train_data(settings)
    
    # Train tokenizer
    tokenization_pipeline(settings)

if __name__ == "__main__":
    main()