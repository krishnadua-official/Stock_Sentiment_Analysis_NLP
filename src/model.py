"""Machine learning model for sentiment analysis"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import logging
from config.settings import MODEL_PATH, VECTORIZER_PATH, RANDOM_STATE, MAX_FEATURES, LABEL_MAP

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Complete sentiment analysis pipeline"""
    
    def __init__(self):
        self.vectorizer = CountVectorizer(
            tokenizer=lambda x: x,  # Tokens already preprocessed
            lowercase=False,
            max_features=MAX_FEATURES
        )
        self.model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        self.is_trained = False
        logger.info("SentimentAnalyzer initialized")
    
    def prepare_features(self, processed_texts):
        """Convert processed texts to feature vectors"""
        # Join tokens back to strings for vectorizer
        text_strings = [' '.join(tokens) for tokens in processed_texts]
        return text_strings
    
    def train(self, processed_texts, labels):
        """Train the sentiment analysis model"""
        logger.info("Starting model training")
        
        # Prepare features
        text_strings = self.prepare_features(processed_texts)
        
        # Vectorize
        X = self.vectorizer.fit_transform(text_strings)
        y = np.array(labels)
        
        # Train model
        self.model.fit(X, y)
        self.is_trained = True
        
        logger.info("Model training completed")
        
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def evaluate(self, processed_texts, labels):
        """Evaluate model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        text_strings = self.prepare_features(processed_texts)
        X = self.vectorizer.transform(text_strings)
        y = np.array(labels)
        
        predictions = self.model.predict(X)
        accuracy = accuracy_score(y, predictions)
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y, predictions, target_names=list(LABEL_MAP.values())))
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'classification_report': classification_report(y, predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(y, predictions)
        }
    
    def predict(self, text_or_tokens):
        """Predict sentiment for new text"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(text_or_tokens, str):
           
            text_strings = [text_or_tokens]
        elif isinstance(text_or_tokens, list) and isinstance(text_or_tokens[0], str):
            # If list of strings
            text_strings = text_or_tokens
        else:
            # If preprocessed tokens
            text_strings = self.prepare_features([text_or_tokens])
        
        X = self.vectorizer.transform(text_strings)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                'prediction': LABEL_MAP[pred],
                'confidence': prob.max(),
                'probabilities': {LABEL_MAP[i]: prob[i] for i in range(len(prob))}
            })
        
        return results[0] if len(results) == 1 else results
    
    def save_model(self):
        """Save trained model and vectorizer"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(VECTORIZER_PATH, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        logger.info(f"Model saved to {MODEL_PATH}")
        logger.info(f"Vectorizer saved to {VECTORIZER_PATH}")
    
    def load_model(self):
        """Load pre-trained model and vectorizer"""
        try:
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(VECTORIZER_PATH, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            self.is_trained = True
            logger.info("Model and vectorizer loaded successfully")
            return True
        except FileNotFoundError:
            logger.warning("No saved model found")
            return False

