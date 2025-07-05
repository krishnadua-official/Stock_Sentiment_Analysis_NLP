import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_handler import DataHandler
from preprocessor import TextPreprocessor
from model import SentimentAnalyzer
from utils import setup_logging, display_data_info, plot_label_distribution, interactive_prediction
import logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description='Financial Sentiment Analysis')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--predict', type=str, help='Predict sentiment for given text')
    parser.add_argument('--interactive', action='store_true', help='Interactive prediction mode')
    parser.add_argument('--show-data', action='store_true', help='Show dataset information')
    
    args = parser.parse_args()
    
    
    data_handler = DataHandler()
    preprocessor = TextPreprocessor()
    analyzer = SentimentAnalyzer()
    
    # Load data
    logger.info("Loading datasets...")
    train_data = data_handler.load_training_data()
    test_data = data_handler.load_test_data()
    
    if args.show_data:
        display_data_info(data_handler)
        return
    
    if args.train:
        logger.info("Starting training pipeline...")
        
        
        train_texts, train_labels = preprocessor.preprocess_dataset(train_data)
        
       
        analyzer.train(train_texts, train_labels)
        
        
        analyzer.save_model()
        
        logger.info("Training completed!")
        
        
        plot_label_distribution(train_labels, "Training Data Label Distribution")
    
    if args.evaluate:
        logger.info("Evaluating model...")
        
        
        if not analyzer.is_trained:
            if not analyzer.load_model():
                logger.error("No trained model found. Please train first.")
                return
        
        
        test_texts, test_labels = preprocessor.preprocess_dataset(test_data)
        
        
        results = analyzer.evaluate(test_texts, test_labels)
        
        
        plot_label_distribution(test_labels, "Test Data Label Distribution")
    
    if args.predict:
        
        if not analyzer.is_trained:
            if not analyzer.load_model():
                logger.error("No trained model found. Please train first.")
                return
        
        
        tokens = preprocessor.preprocess_text(args.predict)
        result = analyzer.predict(tokens)
        
        print(f"\nText: '{args.predict}'")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
    
    if args.interactive:
        # Load model if not already trained
        if not analyzer.is_trained:
            if not analyzer.load_model():
                logger.error("No trained model found. Please train first.")
                return
        
        interactive_prediction(analyzer, preprocessor)
    
    if not any([args.train, args.evaluate, args.predict, args.interactive, args.show_data]):
        parser.print_help()

if __name__ == "__main__":
    main()
