from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
import os 

def main():
    
    
    # Iterate over all files in path and its subfolders
    path = './runs/'
    for root, dirs, files in os.walk(path):
        for file in files:
            experiment = tb.data.experimental.ExperimentFromDev(file)
            df = experiment.get_scalars()
            df

    

if __name__ == "__main__":
    main()