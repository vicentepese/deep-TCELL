# Load libraries
import torch 
import pandas as pd 
import numpy as np
import json
import tokenizers
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns


from torch import tensor
from torch.utils.data import DataLoader, Dataset
from transformers import  RobertaConfig
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers import pre_tokenizers, normalizers, Tokenizer
from tokenizers.normalizers import Lowercase, NFD
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from utils.utils import prob2label
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from tqdm import tqdm

# Create dataset
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
        if isinstance(self.tokenizer.model, tokenizers.models.WordLevel):
            item = {
                "CDR3a": self.data.CDR3a[index],
                "CDR3b": self.data.CDR3b[index]
                }
        # if isinstance(self.tokenizer.model, tokenizers.models.WordLevel):
        #     self.tokenizer.enable_padding(length=self.max_len)
        #     CDR3a = " ".join(list(self.data.CDR3a[index]))
        #     CDR3b = " ".join(list(self.data.CDR3b[index]))
        #     encodings_CDR3a = self.tokenizer.encode(CDR3a)
        #     encodings_CDR3b = self.tokenizer.encode(CDR3b)
        #     item = {
        #         "ids_CDR3a":tensor(encodings_CDR3a.ids, dtype=torch.long),
        #         "ids_CDR3b":tensor(encodings_CDR3b.ids, dtype=torch.long),
        #         "attention_mask_CDR3a": tensor(encodings_CDR3a.attention_mask, dtype=torch.long),
        #         "attention_mask_CDR3b": tensor(encodings_CDR3b.attention_mask, dtype=torch.long),
        #         "CDR3a": self.data.CDR3a[index],
        #         "CDR3b": self.data.CDR3b[index]
        #         }
            
        elif isinstance(self.tokenizer.model, tokenizers.models.BPE):
            self.tokenizer.enable_padding(length=self.max_len)
            encodings_CDR3a = self.tokenizer.encode(self.data.CDR3a[index]) 
            encodings_CDR3b = self.tokenizer.encode(self.data.CDR3b[index]) 
            item = {
                "ids_CDR3a":tensor(encodings_CDR3a.ids, dtype=torch.long),
                "ids_CDR3b":tensor(encodings_CDR3b.ids, dtype=torch.long),
                "attention_mask_CDR3a": tensor(encodings_CDR3a.attention_mask, dtype=torch.long),
                "attention_mask_CDR3b": tensor(encodings_CDR3b.attention_mask, dtype=torch.long),
                "CDR3a": self.data.CDR3a[index],
                "CDR3b": self.data.CDR3b[index]
                }
        if self.label == "multilabel":
            item["target"] = tensor(self.data[["activatedby_HA", "activatedby_NP", "activatedby_HCRT"]].iloc[index], dtype=torch.long)
            item["label"] = self.data["activated_by"].iloc[index]
        else:
            item["target"] = [self.data[self.label][index]]
            item["label"] = self.data["activated_by"].iloc[index]
        return item

    def __len__(self):
        return len(self.data)

def main():
    
    # Load settings
    with open("settings.json","r") as inFile:
        settings = json.load(inFile)
        
    # Load train and test dataset and merge
    train_dataset = pd.read_csv(settings['file']['train_data'])
    test_dataset = pd.read_csv(settings['file']['test_data'])
    dataset = pd.concat([train_dataset, test_dataset], axis=0)

    # Set device 
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load model
    model = torch.load('best_model_test')
    model.to(device)
    model.eval()
    
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
    loader_params = {'batch_size': 1,
                'shuffle': False,
                'num_workers': 0
                }
    train_dataloader = DataLoader(train_data, **loader_params)
    test_dataloader = DataLoader(test_data, **loader_params)

    # Get list of amino acids 
    AA_vocab = [key for key in tokenizer.get_vocab().keys() if "[" not in key]
    
    # Initialize model and tokenizer
    tokenizer = test_data.tokenizer
    tokenizer.enable_padding(length=test_data.max_len)
    model.eval()

    # Output dictionaries
    HA_alpha_dict, NP_alpha_dict, HCRT_alpha_dict, neg_alpha_dict =  defaultdict(pd.DataFrame), defaultdict(pd.DataFrame), defaultdict(pd.DataFrame), defaultdict(pd.DataFrame)
    HA_beta_dict, NP_beta_dict, HCRT_beta_dict, neg_beta_dict =  defaultdict(pd.DataFrame), defaultdict(pd.DataFrame), defaultdict(pd.DataFrame), defaultdict(pd.DataFrame)

    # Main loop
    for data in tqdm(test_dataloader):
        CDR3a, CDR3b = data["CDR3a"][0], data["CDR3b"][0]
        targets = data["target"].tolist()[0]
        label = data["label"][0]
        target_idx = np.where(targets == 1)
        
        # Compute original output 
        encodings_CDR3a = torch.unsqueeze(tensor(tokenizer.encode(" ".join(CDR3a)).ids, dtype=torch.long), dim=0).to(device)
        encodings_CDR3b = torch.unsqueeze(tensor(tokenizer.encode(" ".join(CDR3b)).ids, dtype=torch.long), dim=0).to(device)
        original_outs = model(encodings_CDR3a, encodings_CDR3b).detach().to('cpu').tolist()[0]
        
        # Alpha chain loop 
        mod_probs, idx_list, AA_list = [], [], []
        for idx, AA_CDR3a in enumerate(CDR3a):
            
            # Remove original AA
            AA_list_CDR3a = AA_vocab.copy()
            AA_list_CDR3a.remove(AA_CDR3a.lower())
            
            # Iterate over all remaining AAs
            for AA in AA_list_CDR3a:
                CDR3a_mod = list(CDR3a)
                CDR3a_mod[idx] = AA.capitalize()
                CDR3a_mod = " ".join(CDR3a_mod)
                encodings_CDR3a_mod =  torch.unsqueeze(tensor(tokenizer.encode(CDR3a_mod).ids, dtype=torch.long), dim=0).to(device)
                mod_outs = model(encodings_CDR3a_mod, encodings_CDR3b).detach().to('cpu').tolist()[0]
                mod_probs += [np.subtract(original_outs, mod_outs).tolist()]
                idx_list += [idx]
                AA_list += [AA]
                
        # Append to dictionaries 
        if label == "activatedby_HA":
            HA_alpha_dict["P_HA"] = pd.concat([HA_alpha_dict["P_HA"], pd.DataFrame({"AA": AA_list, "position":idx_list, "delta_prob":[item[0] for item in mod_probs]})], axis=0)
            HA_alpha_dict["P_NP"] = pd.concat([HA_alpha_dict["P_NP"], pd.DataFrame({"AA": AA_list, "position":idx_list, "delta_prob":[item[1] for item in mod_probs]})], axis=0)
            HA_alpha_dict["P_HCRT"] = pd.concat([HA_alpha_dict["P_HCRT"], pd.DataFrame({"AA": AA_list, "position":idx_list, "delta_prob":[item[2] for item in mod_probs]})], axis=0)
        elif label == "activatedby_NP":
            NP_alpha_dict["P_HA"] = pd.concat([NP_alpha_dict["P_HA"], pd.DataFrame({"AA": AA_list, "position":idx_list, "delta_prob":[item[0] for item in mod_probs]})], axis=0)
            NP_alpha_dict["P_NP"] = pd.concat([NP_alpha_dict["P_NP"], pd.DataFrame({"AA": AA_list, "position":idx_list, "delta_prob":[item[1] for item in mod_probs]})], axis=0)
            NP_alpha_dict["P_HCRT"] = pd.concat([NP_alpha_dict["P_HCRT"], pd.DataFrame({"AA": AA_list, "position":idx_list, "delta_prob":[item[2] for item in mod_probs]})], axis=0)
        elif label == "activatedby_HCRT":
            HCRT_alpha_dict["P_HA"] = pd.concat([HCRT_alpha_dict["P_HA"], pd.DataFrame({"AA": AA_list, "position":idx_list, "delta_prob":[item[0] for item in mod_probs]})], axis=0)
            HCRT_alpha_dict["P_NP"] = pd.concat([HCRT_alpha_dict["P_NP"], pd.DataFrame({"AA": AA_list, "position":idx_list, "delta_prob":[item[1] for item in mod_probs]})], axis=0)
            HCRT_alpha_dict["P_HCRT"] = pd.concat([HCRT_alpha_dict["P_HCRT"], pd.DataFrame({"AA": AA_list, "position":idx_list, "delta_prob":[item[2] for item in mod_probs]})], axis=0)
        else:
            neg_alpha_dict["P_HA"] = pd.concat([neg_alpha_dict["P_HA"], pd.DataFrame({"AA": AA_list, "position":idx_list, "delta_prob":[item[0] for item in mod_probs]})], axis=0)
            neg_alpha_dict["P_NP"] = pd.concat([neg_alpha_dict["P_NP"], pd.DataFrame({"AA": AA_list, "position":idx_list, "delta_prob":[item[1] for item in mod_probs]})], axis=0)
            neg_alpha_dict["P_HCRT"] = pd.concat([neg_alpha_dict["P_HCRT"], pd.DataFrame({"AA": AA_list, "position":idx_list, "delta_prob":[item[2] for item in mod_probs]})], axis=0)
                    
                
            

    
if __name__ == "__main__":
    main()