import torch 
import transformers 
from transformers.models.roberta import *
from torch import nn
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings 



class Net(nn.modules):
    
    def __init__(self, n_labels:int, model_config:RobertaConfig, classifier_drouput:float) -> None:
        super(Net, self).__init__()
        
        self.n_labels = n_labels
        self.config = model_config
        self.classifier_dropout = classifier_drouput
        
        RobertaEmbeddings