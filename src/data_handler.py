import pandas as pd
import requests
from io import StringIO
import logging
from pathlib import Path
from config.settings import TRAIN_DATA_URL, TEST_DATA_URL, DATA_DIR

logger = logging.getLogger(__name__)

class DataHandler:
    """Handles data loading, saving, and basic operations"""
    
    def __init__(self):
        self.train_data = None
        self.test_data = None
    
    def load_data_from_url(self, url):
        """Load data from URL"""
        try:
            logger.info(f"Loading data from: {url}")
            response = requests.get(url)
            response.raise_for_status()
            data = pd.read_csv(StringIO(response.text))
            logger.info(f"Successfully loaded {len(data)} samples")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def load_data_from_file(self, filepath):
        """Load data from local file"""
        try:
            data = pd.read_csv(filepath)
            logger.info(f"Loaded {len(data)} samples from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading file {filepath}: {e}")
            raise
    
    def save_data(self, data, filename):
        """Save data to local file"""
        filepath = DATA_DIR / filename
        data.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
    
    def load_training_data(self):
        """Load training data"""
        train_file = DATA_DIR / "train_data.csv"
        
        if train_file.exists():
            self.train_data = self.load_data_from_file(train_file)
        else:
            self.train_data = self.load_data_from_url(TRAIN_DATA_URL)
            self.save_data(self.train_data, "train_data.csv")
        
        return self.train_data
    
    def load_test_data(self):
        """Load test data"""
        test_file = DATA_DIR / "test_data.csv"
        
        if test_file.exists():
            self.test_data = self.load_data_from_file(test_file)
        else:
            self.test_data = self.load_data_from_url(TEST_DATA_URL)
            self.save_data(self.test_data, "test_data.csv")
        
        return self.test_data
    
    def get_data_info(self, data):
        """Get basic information about the dataset"""
        info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'label_distribution': data['Label'].value_counts().to_dict() if 'Label' in data.columns else None,
            'sample_texts': data['Sentence'].head().tolist() if 'Sentence' in data.columns else None
        }
        return info

