
import numpy as np 
import pandas as pd 
import json
import tokenizers
import torch
from models.roberta_multilabel import Net
from utils.utils import *
import argparse
from tqdm import tqdm
from operator import add
from collections import OrderedDict, defaultdict

from torch.utils.data import DataLoader, Dataset
from torch import nn, tensor
from torch.utils.tensorboard import SummaryWriter

from transformers import  RobertaConfig
from tokenizers.models import WordLevel, BPE
from tokenizers import Tokenizer
from sklearn.metrics import recall_score, precision_score
   
class CDR3Dataset(Dataset):
    
    def __init__(self, settings:dict, train:bool = True, label:str = None, tokenizer:tokenizers.Tokenizer=None, equal:bool=False) -> None:
        cols = ["activatedby_HA", "activatedby_NP", "activatedby_HCRT", "activated_any", "multilabel", "negative"]
        if label not in cols:
            raise ValueError("Invalid label type. Expected one of %s" % cols)
        else: 
            self.label = label
        if equal and label == "num_label":
            raise ValueError("Equal size sets only allowed for binary classifications. num_label is multiclass.")
        
        if train == True:
            path_to_data = settings["file"]["train_data"] 
        else:
            path_to_data = settings["file"]["test_data"]   
              
        self.path_to_data = path_to_data
        self.data = pd.read_csv(self.path_to_data)
        if equal == True:
            min_sample=np.min(self.data[self.label].value_counts()) 
            data_pos = self.data[self.data[self.label]==1].sample(min_sample)
            data_neg = self.data[self.data[self.label]==0].sample(min_sample)
            self.data = pd.concat([data_pos, data_neg], ignore_index=True)
        
        if label == "multilabel":
            self.labels = [0,1]
            self.n_labels = 3
        else:
            self.labels = np.unique(self.data[[self.label]])
            self.n_labels = len(self.labels)
            
        self.max_len = self.data.CDR3ab.str.len().max()
        
        self.tokenizer = tokenizer
        
    def __getitem__(self, index:int):
        if isinstance(self.tokenizer.model, tokenizers.models.WordLevel):
            self.tokenizer.enable_padding(length=self.max_len)
            CDR3ab = " ".join(list(self.data.CDR3ab[index]))
            encodings = self.tokenizer.encode(CDR3ab)
            item = {
                "ids":tensor(encodings.ids, dtype=torch.long),
                "attention_mask": tensor(encodings.attention_mask, dtype=torch.long)
                }
        elif isinstance(self.tokenizer.model, tokenizers.models.BPE):
            self.tokenizer.enable_padding(length=self.max_len)
            encodings = self.tokenizer.encode(self.data.CDR3ab[index]) 
            item = {
                "ids":tensor(encodings.ids, dtype=torch.long),
                "attention_mask": tensor(encodings.attention_mask, dtype=torch.long)
                }
        if self.label == "multilabel":
            item["target"]=tensor(self.data[["activatedby_HA", "activatedby_NP", "activatedby_HCRT"]].iloc[index],dtype =torch.long)
        else:
            item["target"] = tensor(self.data[self.label][index], dtype=torch.long)
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
            settings['param']['learning_rate'] =  np.random.choice(settings['opt_param']['learning_rate'],1).item()
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

    # Create tonekizer from tokenizers library 
    if settings["param"]["tokenizer"] == "BPE":
        tokenizer = Tokenizer(BPE()).from_file(settings["tokenizer"]["BPE"])
    elif settings["param"]["tokenizer"] == "WL":
        tokenizer = Tokenizer(WordLevel()).from_file(settings["tokenizer"]["WL"])
    else:
        raise ValueError("Unknown tokenizer. Tokenizer argument must be BPE or WL.")
    tokenizer.enable_padding()
        
    # Create training and test dataset
    dataset_params={"label":settings["database"]["label"], "tokenizer":tokenizer}
    train_data = CDR3Dataset(settings,train=True, equal=False, **dataset_params)
    test_data =CDR3Dataset(settings, train=False, **dataset_params)
    
    # Crate dataloaders
    loader_params = {'batch_size': settings["param"]['batch_size'],
                'shuffle': True,
                'num_workers': 0
                }
    train_dataloader = DataLoader(train_data, **loader_params)
    test_dataloader = DataLoader(test_data, **loader_params)
    
    model_config = RobertaConfig(vocab_size = tokenizer.get_vocab_size(),
                               problem_type="multi_label_classification",
                               hidden_dropout_prob=settings['param']['dropout'],
                               **settings['model_config'])

    
    # Create the model and add to
    model = Net(n_labels=train_data.n_labels, model_config=model_config, classifier_dropout=settings['param']['dropout'])
    model.to(device)
    
    # Add to model and hyperparameter to writer
    item = next(iter(train_dataloader))
    writer.add_graph(model, [item['ids'].to(device), item['attention_mask'].to(device)])
    
    # Initialize model weights
    model.apply(init_weights)
    
    # Create the loss function and optimizer
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr = settings["param"]['learning_rate'])
        
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
            loss = loss_function(output, targets.to(torch.float32))
            
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
        for label, index in zip(["HA", "NP", "HCRT"], range(4)):
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
            loss = loss_function(output, targets.to(torch.float32))
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
        for label, index in zip(["HA", "NP", "HCRT"], range(4)):
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