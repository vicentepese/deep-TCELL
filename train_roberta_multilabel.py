
import numpy as np 
import pandas as pd 
import json
import tokenizers 
import transformers
import torch
from models.roberta_multilabel import Net

from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from torch import nn, tensor

from transformers import  RobertaConfig, RobertaModel
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers import pre_tokenizers
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD
from tokenizers.pre_tokenizers import ByteLevel

from sklearn.metrics import recall_score, precision_score

def init_weights(layer:torch.nn) -> torch.nn.Linear:
    """init_weights [Initializes weight of Linear layers of he model]

    Args:
        layer (torch.nn): [Layer of the model]. Defaults to False.
    """
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform(layer.weight)
        layer.bias.data.fill_(0.01)
        
        
def prob2label(output:torch.tensor, threshold:float) -> torch.tensor:
    """prob2label [summary]

    Args:
        output (torch.tensor): [Tensor containing the probability for each label]
        threshold (float): [Threshold to consider the prediction. Must a value between 0 and 1]

    Raises:
        ValueError: [If threshold is not between 0 and 1]

    Returns:
        np.array: [Binary array with labels]
    """
    if threshold > 1 or threshold < 0:
        raise ValueError("Threshold must be bigger than 0 and smaller than 1")
        
    out_label = []
    for sample in output:
        out_label.append([1 if prob > threshold else 0 for prob in sample])
    return tensor(out_label)

def multilabelaccuracy(out_label:torch.tensor, targets:torch.tensor) -> np.array:
    """multilabelaccuracy [summary]

    Args:
        out_label (torch.tensor): [Binary array with labels]
        targets (torch.tensor): [Binary targets]

    Returns:
        np.array: [Percentage of correctly labeled targets]
    """
    
    return torch.sum(out_label==targets)/(np.sum([len(target) for target in targets]))
    
def get_recall_precision(y_true, y_pred) -> list:
    """get_recall [Computes recall for each of the labels (columns)]

    Args:
        y_true ([type]): [True labels / targets]
        y_pred ([type]): [Predicted labels]

    Returns:
        np.array: [description]
    """
    y_true = list(map(list, zip(*y_true.tolist())))
    y_pred = list(map(list, zip(*y_pred.tolist())))
    
    recall, precision = [], []
    for i in range(len(y_true)):
        recall.append(recall_score(y_true[i], y_pred[i], zero_division=0))
        precision.append(precision_score(y_true[i], y_pred[i], zero_division=0))
    return recall, precision
        
            
class CDR3Dataset(Dataset):
    
    def __init__(self, settings:dict, train:bool = True, label:str = None, tokenizer:tokenizers.Tokenizer=None, equal:bool=False) -> None:
        cols = ["num_label", "activatedby_HA", "activatedby_NP", "activatedby_HCRT", "activated_any", "multilabel"]
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
    
    # Set device 
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Set random seed
    seed_nr = 1964
    torch.manual_seed(seed_nr)
    np.random.seed(seed_nr)
  
    # Create tonekizer from tokenizers library 
    normalizer = normalizers.Sequence([Lowercase(), NFD()])
    pre_tokenizer = pre_tokenizers.Sequence([ByteLevel()])
    tokenizer = ByteLevelBPETokenizer(settings["file"]["tokenizer_vocab"], settings["file"]["tokenizer_merge"])
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer

    # Create training and test dataset
    dataset_params={"label":settings["database"]["label"], "tokenizer":tokenizer}
    train_data = CDR3Dataset(settings,train=True, equal=False, **dataset_params)
    test_data =CDR3Dataset(settings, train=False, **dataset_params)
    
    # Crate dataloaders
    loader_params = {'batch_size': settings["param"]["batch_size"],
                'shuffle': True,
                'num_workers': 0
                }
    train_dataloader = DataLoader(train_data, **loader_params)
    test_dataloader = DataLoader(test_data, **loader_params)
    
    model_config = RobertaConfig(vocab_size = 4050,
                                hidden_size = 768,
                                num_attention_heads = 12,
                                num_hidden_layers = 12,
                                problem_type="multi_label_classification",
                                hidden_dropout_prob=0.1)
    
    # Create the model 
    model = Net(n_labels=train_data.n_labels, model_config=model_config)
    model.to(device)
    
    # Initialize model weights
    model.apply(init_weights)
    
    # Create the loss function and optimizer
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr = settings["param"]["learning_rate"])
        
    def calcuate_accu(big_idx, targets):
        n_correct = (big_idx==targets).sum().item()
        return n_correct
    
    # Training routine 
    for epoch in tqdm(range(settings["param"]["n_epochs"])):
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
            output=model(ids, attention_mask)
            loss = loss_function(output, targets.to(torch.float32))
            
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
                big_val, big_idx = torch.max(output.data, dim=1)
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
                print("Training recall for " + label + " " + str(np.round(precision_label, decimals=3)))
        

        
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