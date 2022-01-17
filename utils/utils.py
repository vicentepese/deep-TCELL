import torch 
import numpy as np 
from torch import nn, tensor
from sklearn.metrics import recall_score, precision_score, accuracy_score

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

def multilabelaccuracy(out_label:list, targets:list) -> np.array:
    """multilabelaccuracy [summary]

    Args:
        out_label (list): [Binary array with labels]
        targets (list): [Binary targets]

    Returns:
        np.array: [Percentage of correctly labeled targets]
    """
    
    # Individual accuracy
    ind_acc = accuracy_score(out_label, targets)
    
    # Overall accuracy
    flat_out = [out for tcr in out_label for out in tcr]
    flat_target = [target for tcr in targets for target in tcr]
    overall_acc = accuracy_score(flat_out, flat_target)
    
    return ind_acc, overall_acc
    
def get_recall_precision(y_true:list, y_pred:list) -> list:
    """get_recall [Computes recall for each of the labels (columns)]

    Args:
        y_true ([list]): [True labels / targets]
        y_pred ([list]): [Predicted labels]

    Returns:
        np.array: [description]
    """
    y_true = list(map(list, zip(*y_true)))
    y_pred = list(map(list, zip(*y_pred)))
    
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