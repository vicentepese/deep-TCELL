import torch 
import transformers 
from transformers.models.roberta import *
from torch import nn, tensor
from transformers.models.roberta.modeling_roberta import * 



class Net(nn.modules):
    
    def __init__(self, n_labels:int, model_config:RobertaConfig, classifier_drouput:float) -> None:
        super(Net, self).__init__()
        
        self.n_labels = n_labels
        self.config = model_config
        self.classifier_dropout = classifier_drouput
        
        self.roberta_encoder = RobertaEncoder(self.config)
        self.l1_out_dim = self.l1.pooler.dense.out_features  
        self.pre_classifier = nn.Linear(self.l1_out_dim,self.l1_out_dim)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.l1_out_dim, self.n_labels)
        
    def forward(self, input_ids:tensor, attention_mask:tensor) -> tensor:
        output_enc = self.roberta_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooler = output_enc.pooler_output
        pooler = self.pre_classifier(pooler)
        pooler = nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = torch.sigmoid(output)
        return output