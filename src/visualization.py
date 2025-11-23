"""
Visualization utilities for data analysis and model evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


def set_style():
    """Set visualization style parameters"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         save_path: str = None) -> None:
    """
    Plot confusion matrix heatmap.
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    save_path : str, optional
        Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true: np.ndarray, 
                  y_pred_proba: np.ndarray,
                  save_path: str = None) -> None:
    """
    Plot ROC curve.
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred_proba : np.ndarray
        Predicted probabilities
    save_path : str, optional
        Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_coefficients(coefficients: dict, 
                             top_n: int = 15,
                             save_path: str = None) -> None:
    """
    Plot top feature coefficients.
    
    Parameters
    ----------
    coefficients : dict
        Dictionary with feature names and coefficients
    top_n : int, default=15
        Number of top features to display
    save_path : str, optional
        Path to save the plot
    """
    features = coefficients['features']
    feature_names = list(features.keys())
    feature_coefs = list(features.values())
    
    # Sort by absolute value
    sorted_idx = np.argsort(np.abs(feature_coefs))[-top_n:]
    
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_coefs = [feature_coefs[i] for i in sorted_idx]
    
    plt.figure(figsize=(10, 8))
    colors = ['red' if x < 0 else 'green' for x in sorted_coefs]
    plt.barh(sorted_names, sorted_coefs, color=colors, alpha=0.7)
    plt.xlabel('Coefficient Value')
    plt.title(f'Top {top_n} Feature Coefficients')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_distribution(data: np.ndarray, 
                             feature_names: list,
                             top_n: int = 6,
                             save_path: str = None) -> None:
    """
    Plot distribution of top features.
    
    Parameters
    ----------
    data : np.ndarray
        Feature data
    feature_names : list
        Names of features
    top_n : int, default=6
        Number of features to display
    save_path : str, optional
        Path to save the plot
    """
    n_features = min(top_n, len(feature_names))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i in range(n_features):
        axes[i].hist(data[:, i], bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_title(feature_names[i])
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
