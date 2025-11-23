"""
Data loading utilities for the Breast Cancer Wisconsin dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw dataset from CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file containing the dataset
        
    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    return pd.read_csv(filepath)


def save_processed_data(data: pd.DataFrame, filepath: str) -> None:
    """
    Save processed dataset to CSV file.
    
    Parameters
    ----------
    data : pd.DataFrame
        Processed dataset to save
    filepath : str
        Path where the dataset should be saved
    """
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def get_data_paths(base_path: str = None) -> dict:
    """
    Get standardized data paths for the project.
    
    Parameters
    ----------
    base_path : str, optional
        Base path for the project. If None, uses current working directory
        
    Returns
    -------
    dict
        Dictionary containing paths to raw and processed data directories
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent
    else:
        base_path = Path(base_path)
    
    return {
        'raw': base_path / 'data' / 'raw',
        'processed': base_path / 'data' / 'processed',
        'models': base_path / 'models',
        'results': base_path / 'results'
    }


def dataset_info(data: pd.DataFrame) -> None:
    """
    Print comprehensive information about the dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset to analyze
    """
    print("Dataset Shape:", data.shape)
    print("\nData Types:")
    print(data.dtypes)
    print("\nMissing Values:")
    print(data.isnull().sum())
    print("\nBasic Statistics:")
    print(data.describe())
