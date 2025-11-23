"""
Model training and prediction utilities
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib


def train_logistic_regression(X_train: np.ndarray, 
                              y_train: np.ndarray,
                              regularization: str = 'l2',
                              C: float = 1.0,
                              max_iter: int = 1000,
                              random_state: int = 42) -> LogisticRegression:
    """
    Train logistic regression model.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training target variable
    regularization : str, default='l2'
        Type of regularization ('l2' or 'l1')
    C : float, default=1.0
        Inverse of regularization strength
    max_iter : int, default=1000
        Maximum number of iterations
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    LogisticRegression
        Trained model
    """
    model = LogisticRegression(
        penalty=regularization,
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs' if regularization == 'l2' else 'liblinear'
    )
    
    model.fit(X_train, y_train)
    
    print(f"Model trained successfully")
    print(f"Regularization: {regularization}, C={C}")
    
    return model


def make_predictions(model: LogisticRegression, 
                    X: np.ndarray, 
                    probability: bool = False) -> np.ndarray:
    """
    Make predictions using trained model.
    
    Parameters
    ----------
    model : LogisticRegression
        Trained model
    X : np.ndarray
        Features for prediction
    probability : bool, default=False
        If True, return probability estimates instead of class predictions
        
    Returns
    -------
    np.ndarray
        Predictions or probabilities
    """
    if probability:
        return model.predict_proba(X)
    else:
        return model.predict(X)


def get_model_coefficients(model: LogisticRegression, 
                          feature_names: list) -> dict:
    """
    Extract and return model coefficients.
    
    Parameters
    ----------
    model : LogisticRegression
        Trained model
    feature_names : list
        Names of features
        
    Returns
    -------
    dict
        Dictionary with feature names and their coefficients
    """
    coefficients = {
        'intercept': model.intercept_[0],
        'features': dict(zip(feature_names, model.coef_[0]))
    }
    return coefficients


def save_model(model: LogisticRegression, filepath: str) -> None:
    """
    Save trained model to file.
    
    Parameters
    ----------
    model : LogisticRegression
        Trained model
    filepath : str
        Path where model should be saved
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> LogisticRegression:
    """
    Load trained model from file.
    
    Parameters
    ----------
    filepath : str
        Path to saved model
        
    Returns
    -------
    LogisticRegression
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
