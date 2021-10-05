import numpy as np 
import pandas as pd 
import json
import tokenizers 
import torch

from torch.utils.data import DataLoader, Dataset

from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers.normalizers import Lowercase, NFD
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import RobertaForSequenceClassification, TrainingArguments, Trainer,RobertaConfig

class CDR3Dataset(Dataset):
    
    def __init__(self, settings:dict, train:bool = True, label:str = None) -> None:
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
        
    def __getitem__(self, index:int):
        CDR3ab, label_idx = self.data[["CDR3ab"]].iloc[index], self.data[[self.label]].iloc[index]
        label = np.zeros(shape = (len(self.labels)))
        label[label_idx] = 1
        return CDR3ab, label
    
    def __len__(self):
        return len(self.data)
    

from transformers import RobertaTokenizer

def main():
    
    # Load settings 
    with open("settings.json", "r") as inFile: 
        settings = json.load(inFile)
        
    # Set random seed
    seed_nr = 1964
    torch.manual_seed(seed_nr)
    np.random.seed(seed_nr)
    
    # Create normalizer and pre-tokenizer
    normalizer = normalizers.Sequence([Lowercase(), NFD()])
    pre_tokenizer = pre_tokenizers.Sequence([ByteLevel()])
    
    # Create Tokenizer (transformers tokenizer giving problems)
    tokenizer = ByteLevelBPETokenizer(vocab=settings["file"]["tokenizer_vocab"], merges=settings["file"]["tokenizer_merge"])
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer

    # Create dataset
    data_train = CDR3Dataset(settings = settings, train=True, label="num_label")
    data_test = CDR3Dataset(settings = settings, train=False, label="num_label")
    
    # Create dataloader
    train_loader = DataLoader(dataset=data_train,
                              batch_size=settings["param"]["batch_size"],
                              shuffle=True)
    test_loader = DataLoader(dataset=data_test,
                            batch_size=settings["param"]["batch_size"],
                            shuffle=True)
    
    # Define model
    model_config = RobertaConfig(
        vocab_size = tokenizer.get_vocab_size(),
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
        output_hidden_states=True
    )
    model = RobertaForSequenceClassification(model_config)
    
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Define Trainer
    trainer_model = Trainer(
        data_collator=tokenizer.encode,
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_test
    )
    
    # Train 
    trainer_model.train()
    trainer_model.save_model(settings["dir"]["Outputs"])
    
    
if __name__ == "__main__":
    main() 