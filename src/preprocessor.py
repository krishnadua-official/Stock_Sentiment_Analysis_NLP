import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import re
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
for data in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.download(data, quiet=True)
    except:
        logger.warning(f"Could not download NLTK data: {data}")

class TextPreprocessor:
    """Handles all text preprocessing operations"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
        logger.info("TextPreprocessor initialized")
    
    def clean_text(self, text):
        """Basic text cleaning"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text"""
        try:
            tokens = word_tokenize(text)
            return tokens
        except:
            return text.split()
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from tokens"""
        return [token for token in tokens if token not in self.stopwords]
    
    def remove_punctuation(self, tokens):
        """Remove punctuation from tokens"""
        return [token for token in tokens if token not in string.punctuation]
    
    def stem_tokens(self, tokens):
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess_text(self, text, remove_stopwords=True, remove_punctuation=True, apply_stemming=True):
        """Complete preprocessing pipeline"""
       
        text = self.clean_text(text)
        
       
        tokens = self.tokenize(text)
        
        
        if remove_punctuation:
            tokens = self.remove_punctuation(tokens)
        
        
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Apply stemming
        if apply_stemming:
            tokens = self.stem_tokens(tokens)
        
        return tokens
    
    def preprocess_dataset(self, df, text_column='Sentence', label_column='Label'):
        """Preprocess entire dataset"""
        logger.info(f"Preprocessing dataset with {len(df)} samples")
        
        processed_texts = []
        labels = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                logger.info(f"Processed {idx} samples")
            
            text = row[text_column]
            processed_tokens = self.preprocess_text(text)
            processed_texts.append(processed_tokens)
            
            if label_column in row:
                labels.append(row[label_column])
        
        logger.info("Preprocessing complete")
        return processed_texts, labels if labels else None

