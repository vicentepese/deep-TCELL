import torch
import transformers

from torch import nn, tensor
from transformers import RobertaModel

class Net(nn.Module):
    
    def __init__(self, n_labels:int, model_config:transformers.RobertaConfig, classifier_dropout=float):
      super(Net, self).__init__()
      self.n_labels = n_labels
      self.config = model_config
      
      self.l1 = RobertaModel(self.config)
      self.l1_out_dim = self.l1.pooler.dense.out_features  
      self.pre_classifier = nn.Linear(self.l1_out_dim,self.l1_out_dim)
      self.dropout = nn.Dropout(classifier_dropout)
      self.classifier = nn.Linear(self.l1_out_dim, self.n_labels)
      
    def forward(self, input_ids:tensor, attention_mask:tensor) -> tensor:
        output_l = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        _ = output_l[0]
        pooler = output_l.pooler_output
        pooler = self.pre_classifier(pooler)
        pooler = nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = torch.sigmoid(output)
        return output 
