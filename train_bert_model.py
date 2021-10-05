import numpy as np 
import pandas as pd 
import json
import tokenizers 
import torch

from torch import Dataset

from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers.normalizers import Lowercase, NFD
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.implementations import ByteLevelBPETokenizer

class CDR3Dataset(Dataset):
    
    def __init__(self, path_to_data:str = None, label:str = None) -> None:
        cols = ["num_label", "activatedby_HA", "activatedby_NP", "activtedby_HCRT", "activatedby_any"]
        if label not in cols:
            raise ValueError("Invalid label type. Expected one of %s" % cols)
        else: 
            self.label = label
        self.path_to_data = path_to_data
        self.data = pd.read_csv(self.path_to_data)
        self.labels = np.unique(self.data[[self.label]])
        
    def __getitem__(self, index:int):
        CDR3ab, label_idx = self.data[["CDR3ab"]].iloc[index], self.data[[self.label]].iloc[index]
        label = np.zeros(shape = (len(self.labels)))
        label[label_idx] = 1
        return CDR3ab, label
    
    def __len__(self):
        return len(self.data)
    

from transformers import RobertaTokenizer

def main():
    
    # Load settings 
    with open("settings.json", "r") as inFile: 
        settings = json.load(inFile)
        
    # Set random seed
    seed_nr = 1964
    torch.manual_seed(seed_nr)
    np.random.seed(seed_nr)
    
    # Create normalizer and pre-tokenizer
    normalizer = normalizers.Sequence([Lowercase(), NFD()])
    pre_tokenizer = pre_tokenizers.Sequence([ByteLevel()])
    
    # Create Tokenizer (transformers tokenizer giving problems)
    tokenizer = ByteLevelBPETokenizer(vocab=settings["file"]["tokenizer_vocab"], merges=settings["file"]["tokenizer_merge"])
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer

    # Create dataset
    
    
    pass

if __name__ == "__main__":
    main() 