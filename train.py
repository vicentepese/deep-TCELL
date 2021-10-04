
from operator import index
import numpy as np 
import pandas as pd
import json

from torch.utils.data import Dataset, DataLoader

class CDR3Dataset(Dataset):
    
    def __init__(self, path_to_data:str = None, label:str = None) -> None:
        cols = ["num_label", "activatedby_HA", "activatedby_NP", "activtedby_HCRT", "activatedby_any"]
        if label not in cols:
            raise ValueError("Invalid label type. Expected one of %s" % cols)
        else: 
            self.label = label
        self.path_to_data = path_to_data
        self.data = pd.read_csv(self.path_to_data)
        
    def __getitem__(self, index:int):
        CDR3ab, label = self.data[["CDR3ab"]].iloc[index], self.data[[self.label]].iloc[index]
        return CDR3ab, label
    
    def __len__(self):
        return len(self.data)
  
def main():
    
    # Load settings 
    with open("settings.json", "r") as inFile:
        settings = json.load(inFile)
        
    # Create Dataset
    cdr3_data = CDR3Dataset(path_to_data=settings["file"]["TCR_data"], label=settings["database"]["label"])
    
    pass

if __name__ == "__main__":
    main()