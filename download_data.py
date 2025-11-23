"""
Data download script for Breast Cancer Wisconsin dataset from Kaggle
"""

import kagglehub
from kagglehub import KaggleDatasetAdapter
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import get_data_paths


def download_dataset():
    """Download Breast Cancer Wisconsin dataset from Kaggle"""
    print("="*60)
    print("Downloading Breast Cancer Wisconsin Dataset")
    print("="*60)
    
    paths = get_data_paths()
    raw_data_path = paths['raw']
    
    # Ensure directory exists
    raw_data_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading to: {raw_data_path}\n")
    
    try:
        # Load the dataset using kagglehub
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "uciml/breast-cancer-wisconsin-data",
            ""
        )
        
        # Save to CSV
        output_file = raw_data_path / 'breast_cancer_data.csv'
        df.to_csv(output_file, index=False)
        
        print(f"✓ Dataset downloaded successfully!")
        print(f"✓ Saved to: {output_file}")
        print(f"✓ Dataset shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}\n")
        
        return df
        
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("\nMake sure you have:")
        print("1. Installed kagglehub: pip install kagglehub")
        print("2. Set up Kaggle API credentials at ~/.kaggle/kaggle.json")
        print("   - Visit https://www.kaggle.com/settings/account for more details")
        sys.exit(1)


if __name__ == "__main__":
    download_dataset()
