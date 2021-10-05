
from numpy import random
from torch.utils.data.dataset import random_split
from model_conv import DeepTCDR
import numpy as np 
import torch
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
        self.labels = np.unique(self.data[[self.label]])
        
    def __getitem__(self, index:int):
        CDR3ab, label_idx = self.data[["CDR3ab"]].iloc[index], self.data[[self.label]].iloc[index]
        label = np.zeros(shape = (len(self.labels)))
        label[label_idx] = 1
        return CDR3ab, label
    
    def __len__(self):
        return len(self.data)
    
def get_dataloaders(settings:dict, dataset:Dataset):
    """get_dataloaders [Get train and test dataloaders]

    Args:
        settings (dict): [Settings dictionary]
        dataset (Dataset): [CDR3 dataset with labels]

    """

    # Get lengths
    lengths = [np.ceil(len(dataset)*(1-settings["param"]["test_split"])).astype('int'),
               np.floor(len(dataset)*settings["param"]["test_split"]).astype('int')]

    # Create random split and dataloaders
    train_dataset, test_dataset = random_split(dataset, lengths)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=settings["param"]["batch_size"],
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=11,
                              shuffle=True)

    return train_dataset, test_dataset, train_loader, test_loader
  
def main():
    
    # Load settings 
    with open("settings.json", "r") as inFile:
        settings = json.load(inFile)
        
    # Set random seed
    seed_nr = 1964
    torch.manual_seed(seed_nr)
    np.random.seed(seed_nr)
    
    # Initialize device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    # Create Dataset
    cdr3_dataset = CDR3Dataset(path_to_data=settings["file"]["TCR_data"], label=settings["database"]["label"])
    
    # Get DatLoaders
    train_dataset, _, train_loader, test_loader = get_dataloaders(settings, cdr3_dataset)
    
    # Import model
    net = DeepTCDR(batch_size=settings["param"]["batch_size"])
    
    # Test with random data 
    x, y = train_dataset[0]
    x = np.random(settings["param"]["batch_size"], 1, 10)
    print(net(x, y))
    
    
    
    
if __name__ == "__main__":
    main()