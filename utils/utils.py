import torch 
import numpy as np 
from torch import nn, tensor
from sklearn.metrics import recall_score, precision_score

def init_weights(layer:torch.nn) -> torch.nn.Linear:
    """init_weights [Initializes weight of Linear layers of he model]

    Args:
        layer (torch.nn): [Layer of the model]. Defaults to False.
    """
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform(layer.weight)
        layer.bias.data.fill_(0.01)
        
        
def prob2label(output:torch.tensor, threshold:float) -> torch.tensor:
    """prob2label [summary]

    Args:
        output (torch.tensor): [Tensor containing the probability for each label]
        threshold (float): [Threshold to consider the prediction. Must a value between 0 and 1]

    Raises:
        ValueError: [If threshold is not between 0 and 1]

    Returns:
        np.array: [Binary array with labels]
    """
    if threshold > 1 or threshold < 0:
        raise ValueError("Threshold must be bigger than 0 and smaller than 1")
        
    out_label = []
    for sample in output:
        out_label.append([1 if prob > threshold else 0 for prob in sample])
    return out_label

def multilabelaccuracy(out_label:torch.tensor, targets:torch.tensor) -> np.array:
    """multilabelaccuracy [summary]

    Args:
        out_label (torch.tensor): [Binary array with labels]
        targets (torch.tensor): [Binary targets]

    Returns:
        np.array: [Percentage of correctly labeled targets]
    """
    
    return torch.sum(out_label==targets)/(np.sum([len(target) for target in targets]))
    
def get_recall_precision(y_true, y_pred) -> list:
    """get_recall [Computes recall for each of the labels (columns)]

    Args:
        y_true ([type]): [True labels / targets]
        y_pred ([type]): [Predicted labels]

    Returns:
        np.array: [description]
    """
    y_true = list(map(list, zip(*y_true.tolist())))
    y_pred = list(map(list, zip(*y_pred.tolist())))
    
    recall, precision = [], []
    for i in range(len(y_true)):
        recall.append(recall_score(y_true[i], y_pred[i], zero_division=0))
        precision.append(precision_score(y_true[i], y_pred[i], zero_division=0))
    return recall, precision

def calcuate_accu(big_idx:int, targets:np.array):
    """calcuate_accu [Calculate accuracy of single-label classification]

    Args:
        big_idx ([type]): [description]
        targets ([type]): [description]

    Returns:
        [type]: [Returns the percentage of single label classification]
    """
    n_correct = (big_idx==targets).sum().item()
    return n_correct