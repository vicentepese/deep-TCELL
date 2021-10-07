from re import M
import numpy as np 
import pandas as pd 
import json
from pandas.io.parsers import read_csv
import tokenizers 
import torch

from train_tokenizer import get_token_train_data, tokenization_pipeline

from torch.utils.data import DataLoader, Dataset
from torch import nn, tensor

from tokenizers import Encoding, normalizers
from tokenizers import pre_tokenizers
from tokenizers.normalizers import Lowercase, NFD
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import DistilBertModel, TrainingArguments, Trainer, RobertaConfig
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
        
        self.tokenizer = tokenizer
        
    def __getitem__(self, index:int):
        encodings = self.tokenizer.encode(self.data.CDR3ab[index])
        item = {
            "ids": tensor(encodings.ids, dtype=torch.long),
            "attention_mask": tensor(encodings.attention_mask,dtype=torch.long),
            "target":tensor(self.data[self.label][index],dtype=torch.long)
        }        
        return item

    def __len__(self):
        return len(self.data)
    
class Net(nn.Module):
    
    def __init__(self, n_labels:int=None):
      super(Net, self).__init__()
      self.n_labels = n_labels
      
      self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
      self.pre_classifier = nn.Linear(768,768)
      self.dropout = nn.Dropout(0.1)
      self.classifier = nn.Linear(768, self.n_labels)
      
    def forward(self, input_ids:tensor, attention_mask:tensor) -> tensor:
        output_l = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_l[0]
        pooler = hidden_state[:,0]
        pooler = self.pre_classifier(pooler)
        pooler = nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
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
    
    # # Create normalizer and pre-tokenizer
    # normalizer = normalizers.Sequence([Lowercase(), NFD()])
    # pre_tokenizer = pre_tokenizers.Sequence([ByteLevel()])
    
    # # Create tokenizer 
    # tokenizer = ByteLevelBPETokenizer(vocab=settings["file"]["tokenizer_vocab"], merges=settings["file"]["tokenizer_merge"])
    # tokenizer.normalizer = normalizer
    # tokenizer.pre_tokenizer = pre_tokenizer
    
    # Create tokenizer
    get_token_train_data(settings)
    tokenizer = tokenization_pipeline(settings)
    tokenizer.enable_truncation(max_length=512)

    # Create training and test dataset
    train_data = CDR3Dataset(settings,train=True, label="num_label", tokenizer=tokenizer)
    test_data =CDR3Dataset(settings, train=False,label="num_label", tokenizer=tokenizer)
    
    # Crate dataloaders
    loader_params = {'batch_size': settings["param"]["batch_size"],
                'shuffle': True,
                'num_workers': 0
                }
    train_dataloader = DataLoader(train_data, **loader_params)
    test_dataloader = DataLoader(test_data, **loader_params)
    
    # Create the model 
    model = Net(n_labels=train_data.n_labels)
    model.to(device)
    
    # Create the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr = settings["param"]["learning_rate"])
    
    # Initialize training routine 
    tr_loss, tst_loss = [], []
    tr_acc, tst_acc = [], []
    n_correct = []
    
    def calcuate_accu(big_idx, targets):
        n_correct = (big_idx==targets).sum().item()
        return n_correct
    
    # Training routine 
    for epoch in range(settings["param"]["n_epochs"]):
        model.train()
        for data in train_dataloader:
            ids = data["ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["target"].to(device)
            
            output=model(ids, attention_mask)
            loss = loss_function(output, targets)
            tr_loss += loss.cpu().detach().numpy()
            big_val, big_idx = torch.max(output.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)
            tr_acc = calcuate_accu(big_idx, targets)/len(targets)
    
    

if __name__ == "__main__":
    main()