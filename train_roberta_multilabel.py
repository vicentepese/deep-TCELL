
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
            
        self.max_len_CDRa = self.data.CDR3a.str.len().max()
        self.max_len_CDRb = self.data.CDR3b.str.len().max()
        self.max_len = max(self.max_len_CDRa, self.max_len_CDRb)
        
        self.tokenizer = tokenizer
        
    def __getitem__(self, index:int):
        
        # Word-Level tokenizing
        if isinstance(self.tokenizer.model, tokenizers.models.WordLevel):
            self.tokenizer.enable_padding(length=self.max_len)
            CDR3a = " ".join(list(self.data.CDR3a[index]))
            CDR3b = " ".join(list(self.data.CDR3b[index]))
            encodings_CDR3a = self.tokenizer.encode(CDR3a)
            encodings_CDR3b = self.tokenizer.encode(CDR3b)
            item = {
                "ids_CDR3a":tensor(encodings_CDR3a.ids, dtype=torch.long),
                "ids_CDR3b":tensor(encodings_CDR3b.ids, dtype=torch.long),
                "attention_mask_CDR3a": tensor(encodings_CDR3a.attention_mask, dtype=torch.long),
                "attention_mask_CDR3b": tensor(encodings_CDR3b.attention_mask, dtype=torch.long)
                }
            
        # BPE tokenizer    
        elif isinstance(self.tokenizer.model, tokenizers.models.BPE):
            self.tokenizer.enable_padding(length=self.max_len)
            encodings_CDR3a = self.tokenizer.encode(self.data.CDR3a[index]) 
            encodings_CDR3b = self.tokenizer.encode(self.data.CDR3b[index]) 
            item = {
                "ids_CDR3a":tensor(encodings_CDR3a.ids, dtype=torch.long),
                "ids_CDR3b":tensor(encodings_CDR3b.ids, dtype=torch.long),
                "attention_mask_CDR3a": tensor(encodings_CDR3a.attention_mask, dtype=torch.long),
                "attention_mask_CDR3b": tensor(encodings_CDR3b.attention_mask, dtype=torch.long)
                }
            
        # Append target    
        if self.label == "multilabel":
            item["target"]=tensor(self.data[["activatedby_HA", "activatedby_NP", "activatedby_HCRT"]].iloc[index],dtype =torch.long)
        else:
            item["target"] = tensor(self.data[self.label][index], dtype=torch.long)
            
        return item

    def __len__(self):
        return len(self.data)

def compute_metrics(metrics:dict, output:list, targets:list) -> dict:
    """compute_metrics [Compute accuracy and individual accuracy, recall, precision]

    Args:
        metrics (dict): [Metrics dictionary]
        output  (list): [List with the output of the epoch converted to labels, (N_samples, N_labels)]
        targets (list): [List with the target of the epoch, (N_samples, N_labels)]
    Returns:
        dict: [Dictionary with the metrics of the epoch]
    """
    
    # Compute accuracies and append 
    indv_acc, overall_acc = multilabelaccuracy(output, targets)
    metrics['acc'] = overall_acc
    metrics['indv_acc'] = indv_acc

    # Compute recall and precision, and append
    recall_epoch, precision_epoch = get_recall_precision(y_true=targets, y_pred=output)
    metrics['recall']  = recall_epoch
    metrics['precision']  = precision_epoch
    
    return metrics

def write_metrics(metrics:dict, writer:SummaryWriter, train:bool, epoch:int) -> None:
    """write_metrics [Write metrics from dictionary to summary writer]

    Args:
        metrics (dict): [Dictionary of metrics]
        writter (SummaryWriter): [Summary writer of the experiment]
        train (bool): [Train metrics dictionnary]
        epoch (int): [Epoch index]
    """
    
    # Train or test type
    type = "train" if train else "test"
    
    # Write metrics
    writer.add_scalar("Loss/" + type + ":", np.mean(metrics['loss']), epoch)
    writer.add_scalar("Accuracy/" + type + ":", metrics['acc'], epoch)
    writer.add_scalar("Individual Accuracy/" + type + ":", metrics['indv_acc'], epoch)
    for label, index in zip(["HA", "NP", "HCRT"], range(3)):
        writer.add_scalar("Recall/" + label + "_" + type, metrics['recall'][index],epoch)
        writer.add_scalar("Precision/" + label + "_" + type, metrics['precision'][index],epoch)

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
    
    # Model configuration
    model_config = RobertaConfig(vocab_size = tokenizer.get_vocab_size(),
                               problem_type="multi_label_classification",
                               hidden_dropout_prob=settings['param']['dropout'],
                               **settings['model_config'])

    # Create the model and move to device
    model = Net(n_labels=train_data.n_labels, model_config=model_config, classifier_dropout=settings['param']['dropout'])
    model.to(device)
    
    # Add to model and hyperparameter to writer
    item = next(iter(train_dataloader))
    writer.add_graph(model, [item['ids_CDR3a'].to(device), item['ids_CDR3b'].to(device)])
    
    # Initialize model weights
    model.apply(init_weights)
    
    # Create the loss function and optimizer
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr = settings["param"]['learning_rate'])
        
    # Training routine 
    max_acc = 0
    metrics_train, metrics_test = defaultdict(list), defaultdict(list)
    for i in tqdm(range(settings["param"]["n_epochs"])):
        model.train()
        target_epoch, out_epoch = [], []
        for data in train_dataloader:
            
            # Prepare data
            ids_CDR3a, ids_CDR3b = data["ids_CDR3a"].to(device), data["ids_CDR3b"].to(device)
            targets = data["target"].to(device)
            
            # Forward pass 
            output = model(input_ids_alpha=ids_CDR3a, input_ids_beta=ids_CDR3b)
            loss = loss_function(output, targets.to(torch.float32))
            
            # Append loss
            metrics_train['loss'].append(loss.cpu().detach().numpy().item())
            
            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Move target to CPU, convert to numpy and append to epoch
            targets = targets.to("cpu").numpy().tolist()
            target_epoch += targets
            
            # Convert probabilities to labels and append to epoch
            out_label = prob2label(output, threshold=0.5)
            out_epoch += out_label
        
        # Compute metrics
        metrics_train = compute_metrics(metrics_train, out_epoch, target_epoch)

        # Add to writer
        write_metrics(metrics_train, writer, train=True, epoch=i)

        # Test 
        model.eval()
        target_epoch, out_epoch = [], []
        for data in test_dataloader:
            
            # Prepare data
            ids_CDR3a, ids_CDR3b = data["ids_CDR3a"].to(device), data["ids_CDR3b"].to(device)
            targets = data["target"].to(device)

            # Forward pass 
            output = model(input_ids_alpha=ids_CDR3a, input_ids_beta=ids_CDR3b)

            # Compute loss
            loss = loss_function(output, targets.to(torch.float32))
            metrics_test['loss'].append(loss.cpu().detach().numpy().item())
            
            # Move target to CPU, convert to numpy and append to epoch
            targets = targets.to("cpu").numpy().tolist()
            target_epoch += targets
            
            # Compoute multi label accuracies and recall, or acc
            out_label = prob2label(output, threshold=0.5)
            out_epoch += out_label
            
        # Compute metrics
        metrics_test = compute_metrics(metrics_test, out_epoch, target_epoch)

        # Add to writer
        write_metrics(metrics_test, writer, train=False, epoch=i)

        # Save model 
        if max_acc < np.max(metrics_train['acc']):
            torch.save(model, 'best_model')
            
    # Add embedding 
    writer.add_embedding(model.l1.embeddings.word_embeddings.weight, metadata=[word for word, val in tokenizer.get_vocab().items()])

    # Add hyperparameter metrics            
    metrics_hp={'training_accuracy':np.max(metrics_train['acc']), 
                'training_loss':np.mean(metrics_train['loss']),
                'test_accuracy':np.mean(metrics_test['acc']), 
                'test_loss':np.mean(metrics_test['loss'])
        
    }
    writer.add_hparams(settings['param'], metrics_hp)
    
    # Flush writer
    writer.flush()
    

if __name__ == "__main__":
    main()