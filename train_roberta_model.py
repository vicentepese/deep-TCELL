import numpy as np 
import pandas as pd 
import json
import tokenizers 
import torch
import os

from torch.utils.data import DataLoader, Dataset

from tokenizers import Encoding, normalizers
from tokenizers import pre_tokenizers
from tokenizers.normalizers import Lowercase, NFD
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import RobertaForSequenceClassification, RobertaConfig, RobertaTokenizer, RobertaForMultipleChoice, TrainingArguments, Trainer

# Test libs
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
        
        self.tokenizer = tokenizer
        
    def __getitem__(self, index:int):
        CDR3ab, label_idx = self.data.CDR3ab[index], self.data[[self.label]].iloc[index]
        label = np.zeros(shape = (len(self.labels)))
        label[label_idx] = 1
        return CDR3ab, label
    
    def __len__(self):
        return len(self.data)
    
class CDR3abEncoders(Dataset):
    
    def __init__(self, encodings:Encoding=None ,labels:list=None, device:str="cpu"):
        self.encodings = encodings
        self.labels = labels
        self.labels_unique = np.unique(self.labels)
        self.device = device
    
    # TODO: bottleneck here in self.encodings, running over it all the time
    def __getitem__(self, idx:int=None) -> dict:
        item = {"input_ids": torch.tensor(self.encodings["input_ids"][idx]).to(self.device),
                "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]).to(self.device)}
        self.label = np.zeros(len(self.labels_unique))
        self.label[self.labels[idx]] = 1
        item["labels"] = torch.tensor(self.label).to(self.device)
        return item
    
    def __len__(self):
        return(int(len(self.labels)))


def main():
    
    # Load settings 
    with open("settings.json", "r") as inFile: 
        settings = json.load(inFile)
    
    # Set device 
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    
    # Set random seed
    seed_nr = 1964
    torch.manual_seed(seed_nr)
    np.random.seed(seed_nr)
        
    # Create normalizer and pre-tokenizer
    normalizer = normalizers.Sequence([Lowercase(), NFD()])
    pre_tokenizer = pre_tokenizers.Sequence([ByteLevel()])
    
    # Create Tokenizer (transformers tokenizer giving problems)
    tokenizer = RobertaTokenizer.from_pretrained(os.path.abspath("tokenizer"))
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    
    # Read dataframe
    train_df = pd.read_csv(settings["file"]["train_data"])
    test_df = pd.read_csv(settings["file"]["test_data"])
    
    # Parse encoders and labels
    train_encodings, train_label = tokenizer.batch_encode_plus(list(train_df.CDR3ab), padding='longest', truncation=True), train_df.num_label
    test_encodings, test_label = tokenizer.batch_encode_plus(test_df.CDR3ab, padding='longest', truncation=True), test_df.num_label
    
    # Create CDR3abEncoding
    data_train_enc = CDR3abEncoders(encodings=train_encodings, labels=train_label, device=device)
    data_test_enc = CDR3abEncoders(encodings=test_encodings, labels=test_label, device=device)

    # # Create dataset
    # data_train = CDR3Dataset(settings = settings, train=True, label="num_label",tokenizer=tokenizer)
    # data_test = CDR3Dataset(settings = settings, train=False, label="num_label",tokenizer=tokenizer)
    
    # Create dataloader
    train_loader = DataLoader(dataset=data_train_enc,
                              batch_size=settings["param"]["batch_size"],
                              shuffle=True)
    test_loader = DataLoader(dataset=data_test_enc,
                            batch_size=settings["param"]["batch_size"],
                            shuffle=True)
    
    # Define model
    model_config = RobertaConfig(
        vocab_size = tokenizer.vocab_size,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        output_hidden_states=True,
        device=device, 
        out_features=6
    )
    model = RobertaForSequenceClassification(model_config).to(device)
    model.classifier.out_proj = torch.nn.Linear(in_features=768, out_features=6, bias=True)
    
    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=settings["dir"]["Outputs"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=settings["param"]["batch_size"],
        per_device_eval_batch_size=settings["param"]["batch_size"],
        learning_rate=settings["param"]["learning_rate"],
        weight_decay=settings["param"]["weight_decay"],
        num_train_epochs=settings["param"]["n_epochs"],
        load_best_model_at_end=True
    )
    
    # Define Trainer
    trainer_model = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train_enc,
        eval_dataset=data_test_enc
    )
    
    # Train 
    trainer_model.train()
    trainer_model.save_model(settings["dir"]["Outputs"])
    
    
if __name__ == "__main__":
    main() 