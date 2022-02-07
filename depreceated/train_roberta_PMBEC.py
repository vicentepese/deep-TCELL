import numpy as np
from numpy.core.fromnumeric import repeat 
import pandas as pd 
import json 
import torch
from torch._C import dtype
from torch.nn.modules import padding
from torch.optim import optimizer
from collections import defaultdict
from utils.utils import *
import argparse
from tqdm import tqdm 
from models.robert_embeddings import Net

from torch.utils.data import DataLoader, Dataset
from torch import nn, tensor 
from torch.utils.tensorboard import SummaryWriter


from transformers import RobertaConfig

from sklearn.metrics import recall_score

class CDR3EmbeddingDataset(Dataset):
    
    def __init__(self, settings:dict, train:bool=True, padding:bool=True) -> None:
        cols = ["activatedby_HA", "activatedby_NP", "activatedby_HCRT", "activated_any", "multilabel", "negative"]
        if train: 
            self.path_to_data = settings["file"]["train_data"]
        else:
            self.path_to_data = settings["file"]["test_data"]
        self.data = pd.read_csv(self.path_to_data)
        self.embeddings_template = pd.read_csv(settings["file"]["PMBEC_matrix"])
        self.n_labels = 4
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
            "target": tensor(self.data[["activatedby_HA", "activatedby_NP", "activatedby_HCRT", "negative"]].iloc[index], 
                            dtype=torch.float64)
        }
        return item 
    
    def __len__(self):
        return len(self.data)
        
def main():
    
    
    # Load settings 
    with open("settings.json", "r") as inFile:
        settings = json.load(inFile)
    
    # Create random sample hyperparameter
    if settings['opt_param']:
        print("Optimizing parameters:")
        if "batch_size" in settings['opt_param']:
            settings['param']['batch_size'] = np.random.choice(settings['opt_param']['batch_size'],1).item()
            print("Batch_size:" + str(settings['param']['batch_size']))
        if "learning_rate" in settings['opt_param']:
            settings['param']['learning_rate'] = np.random.uniform(settings['opt_param']['learning_rate'][0],
                                                                   settings['opt_param']['learning_rate'][1])
            print("Learning rate: " + str(settings['param']['learning_rate']))
        if "dropout" in settings['opt_param']:
            settings['param']['dropout'] =  np.random.choice(settings['opt_param']['dropout'],1).item()
            print("Dropout: " + str(settings['param']['dropout']))

        # Parse arguments for optimization
        parser = argparse.ArgumentParser(description='Optimization parameters')
        parser.add_argument('--jobid', type = str, help="slurmjobid", default="")

        # Execute the parse_args() method
        args = parser.parse_args()
        print(args.jobid)
        
        # Initialize tensorboard session
        writer = SummaryWriter(settings['dir']['runs'] + str(args.jobid))
    else:
        writer = SummaryWriter(settings['dir']['runs'])    
    
    # Set device 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: " + device)
    
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
        attention_probs_dropout_prob=settings['param']['dropout'],
        hidden_dropout_prob=settings['param']['dropout'],
        vocab_size=21,
        **settings['model_config']
    )
    
    # Create model 
    model = Net(n_labels=train_data.n_labels,
                model_config=model_config, 
                classifier_drouput=settings['param']['dropout'])
    model.to(device)
    model = model.double()
    
    # Add to model and hyperparameter to writer
    item = next(iter(train_dataloader))
    writer.add_graph(model, [item['ids'].to(device), item['attention_mask'].to(device)])
    
    # Initialize weights
    model.apply(init_weights)
    
    # Create loss function and optimizer 
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=settings["param"]["learning_rate"])
    
    # Training routine 
    max_acc = 0
    for i in tqdm(range(settings["param"]["n_epochs"])):
        model.train()
        metrics_epoch = defaultdict(list)
        for data in train_dataloader:
            
            # Prepare data
            ids = data["ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["target"].to(device)
            
            # Forward pass 
            output = model(ids, attention_mask)
            loss = loss_function(output, targets.to(torch.float64))
            
            # Compute loss
            metrics_epoch['tr_loss'] += [loss.cpu().detach().numpy()]
            
            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Compoute multi label accuracies and recall, or acc
            out_label = prob2label(output, threshold=0.5)
            metrics_epoch['tr_acc'] += [multilabelaccuracy(out_label, targets.to("cpu"))]
            recall_epoch, precision_epoch = get_recall_precision(y_true=targets.to("cpu"), y_pred=out_label)
            metrics_epoch['tr_recall'].append(recall_epoch)
            metrics_epoch['tr_precision'].append(precision_epoch)

        # Add to writer
        writer.add_scalar("Loss/train:", np.mean(metrics_epoch['tr_loss']), i)
        writer.add_scalar("Accuracy/train:", np.mean(metrics_epoch['tr_acc']), i)
        for label, index in zip(["HA", "NP", "HCRT", 'negative'], range(4)):
            recall_label = np.mean([val[index] for val in metrics_epoch['tr_recall']])
            precision_label = np.mean([val[index] for val in metrics_epoch['tr_precision']])
            writer.add_scalar("Recall/" + label + "_train", recall_label,i)
            writer.add_scalar("Precision/" + label + "_train", precision_label,i)

                
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
            loss = loss_function(output, targets.to(torch.float64))
            metrics_epoch['tst_loss'] += [loss.cpu().detach().numpy()]
            
            # Compoute multi label accuracies
            out_label = prob2label(output, threshold=1/len(output[0]))
            metrics_epoch['tst_acc'] += [multilabelaccuracy(out_label, targets.to("cpu"))]
            recall_epoch, precision_epoch =get_recall_precision(y_true=targets.to("cpu"), y_pred=out_label)
            metrics_epoch['tst_recall'].append(recall_epoch)
            metrics_epoch['tst_precision'].append(precision_epoch)


        # Add to writer
        writer.add_scalar("Loss/test:", np.mean(metrics_epoch['tst_loss']), i)
        writer.add_scalar("Accuracy/test:", np.mean(metrics_epoch['tst_acc']), i)
        for label, index in zip(["HA", "NP", "HCRT", 'negative'], range(4)):
            recall_label = np.mean([val[index] for val in metrics_epoch['tst_recall']])
            precision_label = np.mean([val[index] for val in metrics_epoch['tst_precision']])
            writer.add_scalar("Recall/" + label + "_test", recall_label, i)
            writer.add_scalar("Precision/" + label + "_test", precision_label, i)

        
        # Save model 
        if max_acc < np.max(metrics_epoch['tr_acc']):
            torch.save(model, 'best_model')
    
    metrics_hp={'tr_acc':np.mean(metrics_epoch['tr_acc']), 
                'tr_loss':np.mean(metrics_epoch['tr_loss']),
                'tst_acc':np.mean(metrics_epoch['tst_acc']), 
                'tst_loss':np.mean(metrics_epoch['tst_loss'])
        
    }
    writer.add_hparams(settings['param'], metrics_hp)

    # Flush writer
    writer.flush()
    

if __name__ == "__main__":
    main()