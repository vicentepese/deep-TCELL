
import numpy as np 
import pandas as pd
import json

from torch.utils.data import Dataset, DataLoader

def main():
    
    # Load settings 
    with open("settings.json", "w") as inFile:
        settings = json.load(inFile)
        
    # Create Database
    
    pass

if __name__ == "__main__":
    main()