import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config.settings import LABEL_MAP
import logging.config
from config.settings import LOGGING_CONFIG

def setup_logging():
    """Setup logging configuration"""
    logging.config.dictConfig(LOGGING_CONFIG)

def display_data_info(data_handler):
    """Display information about loaded datasets"""
    logger = logging.getLogger(__name__)
    
    if data_handler.train_data is not None:
        train_info = data_handler.get_data_info(data_handler.train_data)
        logger.info("Training Data Info:")
        logger.info(f"Shape: {train_info['shape']}")
        logger.info(f"Label distribution: {train_info['label_distribution']}")

    if data_handler.test_data is not None:
        test_info = data_handler.get_data_info(data_handler.test_data)
        logger.info("Test Data Info:")
        logger.info(f"Shape: {test_info['shape']}")
        if test_info['label_distribution']:
            logger.info(f"Label distribution: {test_info['label_distribution']}")


def plot_label_distribution(labels, title="Label Distribution"):
    """Plot distribution of sentiment labels"""
    plt.figure(figsize=(8, 6))
    
    
    unique, counts = np.unique(labels, return_counts=True)
    label_names = [LABEL_MAP[label] for label in unique]
    
    plt.bar(label_names, counts, alpha=0.7)
    plt.title(title)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        plt.text(i, count + max(counts) * 0.01, str(count), ha='center')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(confusion_matrix, title="Confusion Matrix"):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    
    labels = list(LABEL_MAP.values())
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def interactive_prediction(analyzer, preprocessor):
    """Interactive prediction interface"""
    logger = logging.getLogger(__name__)
    
    if not analyzer.is_trained:
        logger.error("Model must be trained first!")
        return
    
    print("Interactive Sentiment Analysis")
    print("Enter financial text to analyze (type 'quit' to exit):")
    
    while True:
        text = input("\n> ").strip()
        
        if text.lower() == 'quit':
            break
        
        if not text:
            continue
        
        try:
            
            tokens = preprocessor.preprocess_text(text)
            
            
            result = analyzer.predict(tokens)
            
            print(f"Text: '{text}'")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("Probabilities:")
            for sentiment, prob in result['probabilities'].items():
                print(f"  {sentiment}: {prob:.4f}")
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")

def validate_input(text, min_length=5):
    """Validate user input"""
    if not text or len(text.strip()) < min_length:
        return False
    return True    
