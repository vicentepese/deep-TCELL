import torch
import transformers

from torch import nn, tensor
from transformers.models.roberta.modeling_roberta import * 

class Net(nn.Module):
    
    def __init__(self, n_labels:int, model_config:transformers.RobertaConfig, classifier_dropout:float):
      super(Net, self).__init__()
      
      # Labels and config
      self.n_labels = n_labels
      self.config = model_config
      
      # Separate embedding layers for alpha and beta chaings
      self.roberta_embeddings_alpha = RobertaEmbeddings(self.config)
      self.roberta_embeddings_beta = RobertaEmbeddings(self.config)
      
      # Roberta Model Backbone 
      self.roberta_encoding = RobertaEncoder(self.config) 
      
      # Preclassifier
      self.pre_classifier = nn.Linear(self.config.hidden_size, 4096)
      self.dropout = nn.Dropout(classifier_dropout)
      
      # Roberta classifier
      self.classifier = nn.Linear(4096, self.n_labels)
      
    def forward(self, input_ids_alpha:tensor, input_ids_beta:tensor) -> tensor:
      
      # Input embeddings
      embeddings_alpha = self.roberta_embeddings_alpha(input_ids=input_ids_alpha)
      embeddings_beta = self.roberta_embeddings_beta(input_ids=input_ids_beta)
      
      # Concat embeddings
      embeddings_cat = torch.cat((embeddings_alpha, embeddings_beta), dim = 1)
      embeddings_shape = embeddings_cat.size()
      batch_size, hidden_size, seq_len = embeddings_shape
      
      # Create extended attention mask 
      attention_mask = torch.ones((batch_size, hidden_size, seq_len), device=embeddings_cat.device)
      
      # Feed to Roberta backbone and pooler
      encodings = self.roberta_encoding(embeddings_cat)   
      encodings = encodings[0]
      encodings = encodings[:,0,:]
      out_pre_classifier = self.pre_classifier(encodings)
      pooler = self.dropout(out_pre_classifier)
      
      # Classifier
      output = self.classifier(pooler)
      output = torch.sigmoid(output)
      
      return output 
