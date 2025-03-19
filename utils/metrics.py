import torch
import numpy as np

def compute_metrics(y_true, y_pred):
    y_true = y_true * 50
    y_pred = y_pred * 50

    abs_error = torch.abs(y_true - y_pred)
    
    square_error = (y_true - y_pred) ** 2
    mad = abs_error.mean().item()
    
    mape = (abs_error / (y_true + 1e-8)).mean().item() * 100
    rmse = torch.sqrt(square_error.mean()).item()

    return mad, mape, rmse

def confusion_matrix(y_true, y_pred, num_classes=None):
    if num_classes is None:
        num_classes = max(y_true.max().item(), y_pred.max().item()) + 1
    
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for t, p in zip(y_true, y_pred):
        conf_matrix[t, p] += 1
        
    return conf_matrix

def calculate_accuracy(y_true, y_pred):
    if isinstance(y_pred, torch.Tensor) and y_pred.dim() > 1 and y_pred.shape[1] > 1:
        y_pred = torch.argmax(y_pred, dim=1)
    
    correct = (y_true == y_pred).sum().item()
    total = y_true.size(0)
    
    return correct / total if total > 0 else 0