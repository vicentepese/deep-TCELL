from operator import index
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
    tcr_df.to_csv(settings["file"]["TCR_data"], header=True, index=False)

    # Split train test and merge
    X_test, X_train = train_test_split(tcr_df, test_size=settings["param"]["test_split"])
    tcr_df_set = {"train":X_train, "test":X_test}
    tcr_df_path = {"train": settings["file"]["train_data"], "test": settings["file"]["test_data"]}
    
    # Tokenize labels , and create individual lables for HA69 and NP136
    for key in tcr_df_set.keys():
        data_pre = tcr_df_set[key]
        le = LabelEncoder().fit(data_pre.activated_by)
        data_pre["num_label"] = le.transform(data_pre.activated_by)
        data_pre["activatedby_HA"] = data_pre.num_label.apply(lambda x: 1 if x in [0,1,2] else 0)
        data_pre["activatedby_HCRT"] = data_pre.num_label.apply(lambda x: 1 if x in [1,3,4] else 0)
        data_pre["activatedby_NP"] = data_pre.num_label.apply(lambda x: 1 if x in [2,4,5] else 0)
        data_pre["negative"] = data_pre.num_label.apply(lambda x: 1 if x == 6 else 0)
        data_pre["activated_any"] = data_pre.num_label.apply(lambda x: 1 if x != 6 else 0)
        data_pre.to_csv(tcr_df_path[key], index=False, header=True)
            
    # Write file to train tokenizer 
    with open(settings["file"]["BPEtokenizer_data"],"w") as outFile:
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
    tokenizer.enable_padding(length=39)
    
    # Train on data 
    tokenizer.train(files=settings["file"]["BPEtokenizer_data"], min_frequency = 2)
    tokenizer.save_model(settings["tokenizer"]["BPE"])
    
    return tokenizer
    
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