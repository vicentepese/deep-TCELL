from operator import index
import numpy as np
import pandas as pd 
import json
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import pre_tokenizers
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

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
    
    # Tokenize labels , and create individual labels for each category
    for key in tcr_df_set.keys():
        data_pre = tcr_df_set[key]
        data_pre.insert(2, "num_label", LabelEncoder().fit_transform(data_pre.activated_by), True)
        data_OHE = pd.get_dummies(data_pre.activated_by, prefix="activated_by")
        data_pre = pd.concat([data_pre, data_OHE], axis=1)
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
    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.enable_padding()
    
    # Create Trainer
    trainer = BpeTrainer(min_frequency=10
                         )
    # Train on data 
    tokenizer.train(files=[settings["file"]["BPEtokenizer_data"]])
    tokenizer.save(settings["tokenizer"]["BPE"], trainder=trainer)
    
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