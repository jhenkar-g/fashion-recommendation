"""Evaluation metrics"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred, num_classes=None):
    y_pred_labels = np.argmax(y_pred, axis=1)
    return {
        'accuracy': accuracy_score(y_true, y_pred_labels),
        'precision': precision_score(y_true, y_pred_labels, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred_labels, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred_labels, average='weighted', zero_division=0)
    }
