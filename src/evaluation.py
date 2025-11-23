"""
Model evaluation metrics and utilities
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     y_pred_proba: np.ndarray = None) -> dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted class labels
    y_pred_proba : np.ndarray, optional
        Predicted probabilities for positive class
        
    Returns
    -------
    dict
        Dictionary containing various evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    # Add AUC-ROC if probabilities are provided
    if y_pred_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def print_evaluation_report(y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           set_name: str = "Test") -> None:
    """
    Print comprehensive evaluation report.
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted class labels
    set_name : str, default="Test"
        Name of the dataset (for display purposes)
    """
    print(f"\n{'='*60}")
    print(f"{set_name} Set Evaluation Metrics")
    print(f"{'='*60}")
    
    metrics = calculate_metrics(y_true, y_pred)
    
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    
    if 'auc_roc' in metrics:
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(cm)
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, 
                              target_names=['Benign', 'Malignant']))


def get_roc_curve(y_true: np.ndarray, 
                 y_pred_proba: np.ndarray) -> tuple:
    """
    Calculate ROC curve values.
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred_proba : np.ndarray
        Predicted probabilities
        
    Returns
    -------
    tuple
        (fpr, tpr, thresholds) - False positive rate, true positive rate, thresholds
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    return fpr, tpr, thresholds
