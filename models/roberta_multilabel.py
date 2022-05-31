from syslog import LOG_EMERG
import sklearn.metrics as metrics
import torch
import torchmetrics
import transformers

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from torch import nn, tensor
from transformers.models.roberta.modeling_roberta import * 

class Net(pl.LightningModule):
  """Net [Roberta model for multilabel classification]

  This model uses two separate embeddings for both alpha and beta chain, followed by a Roberta backbone 
  defined in model_settings. The model uses a Sigmoid activation function in the classifier to provide 
  independent probabilities of T cell activation.
  
  """
    
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
    
    # Loss
    self.loss = torch.nn.BCELoss()
    
    # Metrics 
    self.metrics = {
      'train_accuracy': torchmetrics.Accuracy(),
      'test_accuracy': torchmetrics.Accuracy(),
      
      'train_precision_micro': torchmetrics.Precision(num_classes=self.n_labels, threshold=0.5, average='micro'),
      'train_precision_macro': torchmetrics.Precision(num_classes=self.n_labels, threshold=0.5, average='macro'),
      'train_precision_samples': torchmetrics.Precision(num_classes=self.n_labels, threshold=0.5, average='samples'),
      'train_precision_all': torchmetrics.Precision(num_classes=self.n_labels, threshold=0.5, average=None),
      
      'test_precision_micro': torchmetrics.Precision(num_classes=self.n_labels, threshold=0.5, average='micro'),
      'test_precision_macro': torchmetrics.Precision(num_classes=self.n_labels, threshold=0.5, average='macro'),
      'test_precision_samples': torchmetrics.Precision(num_classes=self.n_labels, threshold=0.5, average='samples'),
      'test_precision_all': torchmetrics.Precision(num_classes=self.n_labels, threshold=0.5, average=None),

      'train_recall_micro': torchmetrics.Recall(num_classes=self.n_labels, threshold=0.5, average='micro'),
      'train_recall_macro': torchmetrics.Recall(num_classes=self.n_labels, threshold=0.5, average='macro'),
      'train_recall_samples': torchmetrics.Recall(num_classes=self.n_labels, threshold=0.5, average='samples'),
      'train_recall_all': torchmetrics.Recall(num_classes=self.n_labels, threshold=0.5, average=None),

      'test_recall_micro': torchmetrics.Recall(num_classes=self.n_labels, threshold=0.5, average='micro'),
      'test_recall_macro': torchmetrics.Recall(num_classes=self.n_labels, threshold=0.5, average='macro'),
      'test_recall_samples': torchmetrics.Recall(num_classes=self.n_labels, threshold=0.5, average='samples'),
      'test_recall_all': torchmetrics.Recall(num_classes=self.n_labels, threshold=0.5, average=None),

      'train_F1_micro': torchmetrics.F1Score(num_classes=self.n_labels, threshold=0.5, average='micro'),
      'train_F1_macro': torchmetrics.F1Score(num_classes=self.n_labels, threshold=0.5, average='macro'),
      'train_F1_samples': torchmetrics.F1Score(num_classes=self.n_labels, threshold=0.5, average='samples'),
      
      'test_F1_micro': torchmetrics.F1Score(num_classes=self.n_labels, threshold=0.5, average='micro'),
      'test_F1_macro': torchmetrics.F1Score(num_classes=self.n_labels, threshold=0.5, average='macro'),
      'test_F1_samples': torchmetrics.F1Score(num_classes=self.n_labels, threshold=0.5, average='samples')
    }
      
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
  
  def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
        return optimizer
  
  def compute_step(self,batch):
        ids_CDR3a, ids_CDR3b, target = batch["ids_CDR3a"], batch["ids_CDR3b"], batch['target']
        label_logits = self.forward(ids_CDR3a, ids_CDR3b)
        threshold = torch.tensor([0.5]).to('cuda') if torch.cuda.is_available() else torch.tensor([0.5])
        label_pred = (label_logits>threshold).float()*1
        return self.loss(label_logits, target.type(torch.float)), target, label_logits, label_pred
      
  def log_metrics(self, type:str, loss:torch.Tensor, target:torch.Tensor, label_logits:torch.Tensor) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.log_dict({'loss':{type:loss}}, on_step=False, on_epoch=True, prog_bar=False)
    self.log_dict({'accuracy':{type:self.metrics[type+'_accuracy'].to(device)(label_logits, target)}}, on_step=False, on_epoch=True, prog_bar=False)
    self.log_dict({'precision/micro':{type:self.metrics[type+'_precision_micro'].to(device)(label_logits, target)}}, on_step=False, on_epoch=True, prog_bar=False)
    self.log_dict({'precision/macro':{type:self.metrics[type+'_precision_macro'].to(device)(label_logits, target)}}, on_step=False, on_epoch=True, prog_bar=False)
    self.log_dict({'precision/samples':{type:self.metrics[type+'_precision_samples'].to(device)(label_logits, target)}}, on_step=False, on_epoch=True, prog_bar=False)
    self.log_dict({'recall/micro':{type:self.metrics[type+'_recall_micro'].to(device)(label_logits, target)}}, on_step=False, on_epoch=True, prog_bar=False)
    self.log_dict({'recall/macro':{type:self.metrics[type+'_recall_macro'].to(device)(label_logits, target)}}, on_step=False, on_epoch=True, prog_bar=False)
    self.log_dict({'recall/samples':{type:self.metrics[type+'_recall_samples'].to(device)(label_logits, target)}}, on_step=False, on_epoch=True, prog_bar=False)
    self.log_dict({'F1/micro':{type:self.metrics[type+'_F1_micro'].to(device)(label_logits, target)}}, on_step=False, on_epoch=True, prog_bar=False)
    self.log_dict({'F1/macro':{type:self.metrics[type+'_F1_macro'].to(device)(label_logits, target)}}, on_step=False, on_epoch=True, prog_bar=False)
    self.log_dict({'F1/samples':{type:self.metrics[type+'_F1_samples'].to(device)(label_logits, target)}}, on_step=False, on_epoch=True, prog_bar=False)
    
    self.log_dict({'precision_all/' + type: {str(idx): val for idx, val in enumerate(self.metrics[type+'_precision_all'].to(device)(label_logits, target))}}, 
                  on_step=False, on_epoch=True, prog_bar=False)
    self.log_dict({'recall_all/' + type: {str(idx): val for idx, val in enumerate(self.metrics[type+'_recall_all'].to(device)(label_logits, target))}}, 
                  on_step=False, on_epoch=True, prog_bar=False)    

  def training_step(self, train_batch):
        loss, target, label_logits, _ = self.compute_step(train_batch)
        self.log_metrics("train", loss, target, label_logits)
        return loss

#   def test_step(self, val_batch):
#         loss, target, label_logits, _ = self.compute_step(val_batch)
#         self.log_metrics("val", loss, target, label_logits)
#         return loss
      
  def validation_step(self, val_batch, batch_idx):
        loss, target, label_logits, _ = self.compute_step(val_batch)
        self.log_metrics("test", loss, target, label_logits)
        return loss  