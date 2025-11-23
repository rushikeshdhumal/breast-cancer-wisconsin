"""
Data preprocessing utilities for breast cancer dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
        
    Returns
    -------
    pd.DataFrame
        Dataset with duplicates removed
    """
    initial_rows = len(data)
    data_clean = data.drop_duplicates()
    duplicates_removed = initial_rows - len(data_clean)
    
    print(f"Removed {duplicates_removed} duplicate rows")
    return data_clean


def handle_missing_values(data: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    strategy : str, default='drop'
        Strategy for handling missing values ('drop' or 'mean')
        
    Returns
    -------
    pd.DataFrame
        Dataset with missing values handled
    """
    missing_count = data.isnull().sum().sum()
    
    if missing_count == 0:
        print("No missing values found")
        return data
    
    print(f"Found {missing_count} missing values")
    
    if strategy == 'drop':
        data_clean = data.dropna()
        print(f"Dropped {len(data) - len(data_clean)} rows with missing values")
    elif strategy == 'mean':
        data_clean = data.fillna(data.mean(numeric_only=True))
        print("Filled missing values with mean")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return data_clean


def separate_features_target(data: pd.DataFrame, target_column: str) -> tuple:
    """
    Separate features and target variable.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    target_column : str
        Name of the target column
        
    Returns
    -------
    tuple
        (X, y) - Features and target variable
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> tuple:
    """
    Standardize features using StandardScaler.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame, optional
        Test features. If None, only X_train is scaled
        
    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, scaler) if X_test is provided
        (X_train_scaled, scaler) if X_test is None
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler


def train_test_split_data(X: pd.DataFrame, y: pd.Series, 
                          test_size: float = 0.2, 
                          random_state: int = 42,
                          stratify: bool = True) -> tuple:
    """
    Split data into training and testing sets.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
    stratify : bool, default=True
        Whether to use stratified split
        
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    stratify_col = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    print(f"Training set size: {len(X_train)} ({100*(1-test_size):.1f}%)")
    print(f"Test set size: {len(X_test)} ({100*test_size:.1f}%)")
    
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(data: pd.DataFrame, 
                        target_column: str,
                        test_size: float = 0.2,
                        random_state: int = 42) -> dict:
    """
    Complete preprocessing pipeline combining all steps.
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw input data
    target_column : str
        Name of the target column
    test_size : float, default=0.2
        Proportion of data for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing train/test splits and scaler
    """
    # Step 1: Remove duplicates
    data = remove_duplicates(data)
    
    # Step 2: Handle missing values
    data = handle_missing_values(data)
    
    # Step 3: Separate features and target
    X, y = separate_features_target(data, target_column)
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Step 5: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }
