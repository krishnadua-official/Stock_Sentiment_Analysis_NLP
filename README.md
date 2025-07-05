This project trains a machine learning model to classify financial text into three sentiment categories:

- **📉 Negative**: Bearish, pessimistic financial sentiment
- **😐 Neutral**: Neutral or factual financial statements  
- **📈 Positive**: Bullish, optimistic financial sentiment


1. **Data Loading**: Downloads financial news datasets from Google Cloud Storage
2. **Text Preprocessing**: 
   - Tokenization (splitting text into words)
   - Stopword removal (getting rid of "the", "and", etc.)
   - Stemming (reducing words to their root form)
3. **Model Training**: Uses Logistic Regression to learn sentiment patterns
4. **Evaluation**: Tests the model and shows accuracy metrics
