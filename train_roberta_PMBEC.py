import numpy as np
from numpy.core.fromnumeric import repeat 
import pandas as pd 
import json 
import torch
from torch._C import dtype
from torch.nn.modules import padding
from torch.optim import optimizer

from utils.utils import *
from tqdm import tqdm 
from models.robert_embeddings import Net

from torch.utils.data import DataLoader, Dataset
from torch import nn, tensor 

from transformers import RobertaConfig

from sklearn.metrics import recall_score

class CDR3EmbeddingDataset(Dataset):
    
    def __init__(self, settings:dict, train:bool=True, padding:bool=True) -> None:
        
        if train: 
            self.path_to_data = settings["file"]["train_data"]
        else:
            self.path_to_data = settings["file"]["test_data"]
        self.data = pd.read_csv(self.path_to_data)
        self.embeddings_template = pd.read_csv(settings["file"]["PMBEC_matrix"])
        self.n_labels = 3
        self.max_len = self.data.CDR3ab.str.len().max()
        self.num_AA = self.embeddings_template.shape[0]
    
    def __getitem__(self, index:int, padding:bool=True) -> dict():
        CDR3ab = list(self.data.CDR3ab[index])
        embeddings = [tensor(self.embeddings_template[AA]) if AA != "_" else torch.zeros(self.num_AA, dtype=torch.float64) for AA in CDR3ab]
        embeddings = torch.stack(embeddings)     
           
        if padding:
            embeddings = torch.cat((embeddings, torch.zeros(self.max_len - embeddings.shape[0], embeddings.shape[1], dtype=torch.float64)))
        
        item ={
            "ids": embeddings,
            "attention_mask": torch.ones(size = embeddings.shape, dtype=torch.float64),
            "target": tensor(self.data[["activatedby_HA", "activatedby_NP", "activatedby_HCRT"]].iloc[index], 
                            dtype=torch.float64)
        }
        return item 
    
    def __len__(self):
        return len(self.data)
        
def main():
    
    
    # Load settings 
    with open("settings.json", "r") as inFile:
        settings = json.load(inFile)
        
    
    # Set device 
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Set random seed
    seed_nr = 1964
    torch.manual_seed(seed_nr)
    np.random.seed(seed_nr)
    
    # Create training and test dataset 
    train_data = CDR3EmbeddingDataset(settings, train = True)
    test_data = CDR3EmbeddingDataset(settings, train=False)
    
    # Create dataloaders
    loader_params = {'batch_size': settings["param"]["batch_size"],
            'shuffle': True,
            'num_workers': 0
            }
    train_dataloader = DataLoader(dataset=train_data, **loader_params)
    test_dataloader = DataLoader(dataset=test_data, **loader_params)
    
    # Create model configuration 
    model_config = RobertaConfig(
        attention_probs_dropout_prob=0.1,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=20,
        max_position_embeddings=512,
        num_attention_heads=10,
        num_hidden_layers=10,
        vocab_size=21
    )
    
    # Create model 
    model = Net(n_labels=train_data.n_labels,
                model_config=model_config, 
                classifier_drouput=0.1)
    model.to(device)
    model = model.double()
    
    # Initialize weights
    model.apply(init_weights)
    
    # Create loss function and optimizer 
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=settings["param"]["learning_rate"])
    
    # Training routine 
    for _ in tqdm(range(settings["param"]["n_epochs"])):
        model.train()
        tr_loss, tst_loss = [], []
        tr_acc, tst_acc = [], []
        tr_recall, tst_recall = [], []
        tr_precision, tst_precision = [], []
        for data in train_dataloader:
            
            # Prepare data
            ids = data["ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["target"].to(device)
            
            # Forward pass 
            output=model(input_ids=ids, attention_mask=attention_mask)
            loss = loss_function(output, targets)
            
            # Compute loss
            tr_loss += [loss.cpu().detach().numpy()]
            
            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Compoute multi label accuracies
            if settings["database"]["label"] == "multilabel":
                out_label = prob2label(output, threshold=1/len(output[0]))
                tr_acc += [multilabelaccuracy(out_label, targets.to("cpu"))]
            else:
                _, big_idx = torch.max(output.data, dim=1)
                n_correct = calcuate_accu(big_idx, targets)
                tr_acc += [(n_correct*100)/targets.size(0)]
            
            # Compute recall and precision
            if settings["database"]["label"] == "multilabel":
                recall_epoch, precision_epoch =get_recall_precision(y_true=targets.to("cpu"), y_pred=out_label)
                tr_recall.append(recall_epoch)
                tr_precision.append(precision_epoch)

        # Verbose
        print("Training Accuracy:" + str(np.mean(tr_acc)))    
        print("Training Loss:" + str(np.mean(tr_loss)))
        
        # Verbose recall and precision
        if settings["database"]["label"] == "multilabel":
            for label, index in zip(["HA", "NP", "HCRT"], range(3)):
                recall_label = np.mean([val[index] for val in tr_recall])
                precision_label = np.mean([val[index] for val in tr_precision])
                print("Training recall for " + label + " " + str(np.round(recall_label, decimals=3)))
                print("Training precision for " + label + " " + str(np.round(precision_label, decimals=3)))
        

        # Test 
        model.eval()
        for data in test_dataloader:
            
            # Prepare data
            ids = data["ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["target"].to(device)

            # Forward pass 
            output=model(ids, attention_mask)

            # Compute loss
            loss = loss_function(output, targets)
            tst_loss += [loss.cpu().detach().numpy()]
            
             # Compoute multi label accuracies
            if settings["database"]["label"] == "multilabel":
                out_label = prob2label(output, threshold=1/len(output[0]))
                tst_acc += [multilabelaccuracy(out_label, targets.to("cpu"))]
            else:
                _, big_idx = torch.max(output.data, dim=1)
                n_correct = calcuate_accu(big_idx, targets)
                tst_acc += [(n_correct*100)/targets.size(0)]
        # Verbose
        print("Test Accuracy:" + str(np.mean(tst_acc)))    
        print("Test Loss:" + str(np.mean(tst_loss)))   
            
    
    

if __name__ == "__main__":
    main()