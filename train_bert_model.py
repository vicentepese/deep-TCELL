from re import M
import numpy as np 
import pandas as pd 
import json
import os
from pandas.io.parsers import read_csv
import tokenizers 
import torch
from tqdm import tqdm
from torch.nn.modules import padding
import transformers

from train_tokenizer import get_token_train_data, tokenization_pipeline

from torch.utils.data import DataLoader, Dataset
from torch import nn, tensor

from tokenizers import Encoding, normalizers
from tokenizers import pre_tokenizers
from tokenizers.normalizers import Lowercase, NFD
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import TrainingArguments, Trainer, RobertaConfig, RobertaTokenizerFast, RobertaModel, RobertaTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer, PreTrainedTokenizerBase

from sklearn.metrics import accuracy_score

class CDR3Dataset(Dataset):
    
    def __init__(self, settings:dict, train:bool = True, label:str = None, tokenizer:tokenizers.Tokenizer=None) -> None:
        cols = ["num_label", "activatedby_HA", "activatedby_NP", "activtedby_HCRT", "activatedby_any"]
        
        if label not in cols:
            raise ValueError("Invalid label type. Expected one of %s" % cols)
        else: 
            self.label = label
        
        if train == True:
            path_to_data = settings["file"]["train_data"] 
        else:
            path_to_data = settings["file"]["test_data"]   
            
        self.path_to_data = path_to_data
        self.data = pd.read_csv(self.path_to_data)
        self.labels = np.unique(self.data[[self.label]])
        self.n_labels = len(self.labels)
        self.max_len = self.data.CDR3ab.str.len().max()
        
        self.tokenizer = tokenizer
        
    def __getitem__(self, index:int):
        encodings = self.tokenizer.encode_plus(self.data.CDR3ab[index],
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True)
        item = {
            "ids": tensor(encodings.input_ids, dtype=torch.long),
            "attention_mask": tensor(encodings.attention_mask,dtype=torch.long),
            "target":tensor(self.data[self.label][index],dtype=torch.long)
        }        
        return item

    def __len__(self):
        return len(self.data)
    
class Net(nn.Module):
    
    def __init__(self, n_labels:int=None, model_config:transformers.RobertaConfig=None):
      super(Net, self).__init__()
      self.n_labels = n_labels
      self.config = model_config
      
      self.l1 = RobertaModel(self.config)
    #   self.l1 = RobertaModel.from_pretrained("roberta-base")
      self.l1_out_dim = self.l1.pooler.dense.out_features  
      self.pre_classifier = nn.Linear(self.l1_out_dim,self.l1_out_dim)
      self.dropout = nn.Dropout(0.1)
      self.classifier = nn.Linear(self.l1_out_dim, self.n_labels)
      
    def forward(self, input_ids:tensor, attention_mask:tensor) -> tensor:
        output_l = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_l[0]
        pooler = output_l.pooler_output
        pooler = self.pre_classifier(pooler)
        pooler = nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        pooler = self.classifier(pooler)
        output = nn.functional.softmax(pooler, dim=1)
        return output 

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
    
    # Create tokenizer 
    tokenizer = RobertaTokenizer.from_pretrained(os.path.abspath("tokenizer"))

    # Create training and test dataset
    train_data = CDR3Dataset(settings,train=True, label="num_label", tokenizer=tokenizer)
    test_data =CDR3Dataset(settings, train=False,label="num_label", tokenizer=tokenizer)
    max_len_seq = train_data.data.CDR3ab.str.len().max()
    
    # Crate dataloaders
    loader_params = {'batch_size': settings["param"]["batch_size"],
                'shuffle': True,
                'num_workers': 0
                }
    train_dataloader = DataLoader(train_data, **loader_params)
    test_dataloader = DataLoader(test_data, **loader_params)
    
    # Create model configuration
    # model_config = RobertaConfig(vocab_size = tokenizer.vocab_size,
    #                              hidden_size = train_data.max_len,
    #                              num_attention_heads = 35,
    #                              num_hidden_layers = 12,
    #                              problem_type="multi_label_classification")
    model_config = RobertaConfig(vocab_size = 3000,
                                hidden_size = 768,
                                num_attention_heads = 12,
                                num_hidden_layers = 12,
                                problem_type="multi_label_classification")
    
    # Create the model 
    model = Net(n_labels=train_data.n_labels, model_config=model_config)
    model.to(device)
    
    # Create the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr = settings["param"]["learning_rate"])
    
    # Initialize training routine 
    tr_loss, tst_loss = [], []
    tr_acc, tst_acc = [], []
    
    def calcuate_accu(big_idx, targets):
        n_correct = (big_idx==targets).sum().item()
        return n_correct
    
    # Training routine 
    for epoch in tqdm(range(settings["param"]["n_epochs"])):
        model.train()
        tr_loss, tst_loss = [], []
        tr_acc, tst_acc = [], []
        for data in train_dataloader:
            
            # Prepare data
            ids = data["ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["target"].to(device)
            
            # Forward pass 
            output=model(ids, attention_mask)
            loss = loss_function(output, targets)
            
            # Compute loss
            tr_loss += [loss.cpu().detach().numpy()]
            
            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Compute accuracies
            big_val, big_idx = torch.max(output.data, dim=1)
            n_correct = calcuate_accu(big_idx, targets)
            tr_acc += [(n_correct*100)/targets.size(0)]
            
        print("Training Accuracy:" + str(np.mean(tr_acc)))    
        print("Training Loss:" + str(np.mean(tr_loss)))    

    

if __name__ == "__main__":
    main()