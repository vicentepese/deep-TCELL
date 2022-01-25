import torch
import transformers

from torch import nn, tensor
from transformers.models.roberta.modeling_roberta import * 

class RobertaBackbone(nn.Module):
    
    def __init__(self, n_labels:int, model_config:transformers.RobertaConfig) -> None:
        super(RobertaBackbone, self).__init__()
        
        # Labels and config
        self.n_labels = n_labels
        self.config = model_config
        
        # Embedding
        self.roberta_embedding = RobertaEmbeddings(self.config)
        
        # Encoder
        self.roberta_encoder = RobertaEncoder(self.config)
        
    def forward(self, input_ids:tensor, attention_mask:tensor) -> tensor:
        
        # Input embeddings
        embeddings = self.roberta_embedding(input_ids=input_ids)
        
        # Encoder
        encodings = self.roberta_encoder(input_ids=embeddings, attention_mask=attention_mask)
        
        return encodings
    

class RobertaBranched(nn.Module):
  """ RobertaBranched [Roberta model for multilabel classification]
  
  The model uses two Roberta backbones, one for each TCR chain, and then concats the representations of each backbone 
  to feed it into the pre-classifier, and classifier. For more info, see roberta_multilabel.py
    
  
  """
    
  def __init__(self, n_labels:int, model_config:transformers.RobertaConfig, classifier_dropout:float):
    super(RobertaBranched, self).__init__()
    
    # Labels and config
    self.n_labels = n_labels
    self.config = model_config
    
    # Alpha bata chain backbone branches 
    self.alpha_branch = RobertaModel(config=model_config, add_pooling_layer=False)
    self.beta_branch = RobertaModel(config=model_config, add_pooling_layer=False)
    
    
    # Preclassifier
    self.pre_classifier = nn.Linear(self.config.hidden_size, 4096)
    self.dropout = nn.Dropout(classifier_dropout)
    
    # Roberta classifier
    self.classifier = nn.Linear(4096, self.n_labels)
      
  def forward(self, input_ids_alpha:tensor, input_ids_beta:tensor) -> tensor:
      
    # Alpha Beta chain encodings 
    alpha_encodings = self.alpha_branch(input_ids_alpha)
    beta_encodings = self.beta_branch(input_ids_beta)
    
    # Concat encodings 
    encodings = torch.cat((alpha_encodings, beta_encodings), dim = 1)
    
    # Pooler
    out_pre_classifier = self.pre_classifier(encodings)
    pooler = self.dropout(out_pre_classifier)
    
    # Classifier
    output = self.classifier(pooler)
    output = torch.sigmoid(output)
    
    return output 
