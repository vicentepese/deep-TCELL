import torch 
import transformers 
from transformers.models.roberta import *
from torch import nn, tensor
from transformers.models.roberta.modeling_roberta import * 



class Net(nn.Module):
    
    def __init__(self, n_labels:int, model_config:RobertaConfig, classifier_drouput:float) -> None:
        super(Net, self).__init__()
        
        self.n_labels = n_labels
        self.config = model_config
        self.classifier_dropout = classifier_drouput
        
        self.roberta_encoder = RobertaEncoder(self.config)
        self.roberta_pooler = RobertaPooler(self.config)
        self.roberta_out_dim = self.roberta_pooler.dense.out_features
        self.pre_classifier = nn.Linear(self.roberta_out_dim,self.roberta_out_dim)
        self.dropout = nn.Dropout(self.classifier_dropout)
        self.classifier = nn.Linear(self.roberta_out_dim, self.n_labels)
        
    def forward(self, input_ids:tensor, attention_mask:tensor) -> tensor:
        output_enc = self.roberta_encoder(input_ids)
        output_enc = output_enc.last_hidden_state
        output_pool = self.roberta_pooler(output_enc)
        pooler = self.pre_classifier(output_pool)
        pooler = nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = torch.sigmoid(output)
        return output