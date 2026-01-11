import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Dict, List
from pathlib import Path
import re
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, max_features: int = 5000, random_state: int = 42):
        """Initialize feature engineering with memory-efficient settings."""
        self.max_features = max_features
        self.random_state = random_state
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            ngram_range=(1, 2),  # Include unigrams and bigrams
            strip_accents='unicode',
            norm='l2'
        )
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def extract_email_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract metadata features from emails."""
        metadata = pd.DataFrame()
        
        # Basic count features
        metadata['token_count'] = df['token_count']
        metadata['message_length'] = df['message'].str.len()
        metadata['cleaned_length'] = df['cleaned_body'].str.len()
        
        # Time-based features (if available)
        if 'timestamp' in df.columns:
            metadata['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            metadata['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            metadata['is_weekend'] = metadata['day_of_week'].isin([5, 6]).astype(int)
        
        # Presence of special characters
        metadata['has_question_mark'] = df['message'].str.contains(r'\?').astype(int)
        metadata['has_exclamation'] = df['message'].str.contains(r'!').astype(int)
        metadata['has_dollar'] = df['message'].str.contains(r'\$').astype(int)
        metadata['url_count'] = df['message'].str.count(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # Email thread features
        metadata['is_reply'] = df['message'].str.contains(r'^re:', case=False).astype(int)
        metadata['is_forward'] = df['message'].str.contains(r'^fw:', case=False).astype(int)
        
        return metadata

    def create_tfidf_features(self, texts: List[str], is_training: bool = True) -> np.ndarray:
        """Create TF-IDF features from text."""
        if is_training:
            tfidf_matrix = self.tfidf.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf.transform(texts)
        return tfidf_matrix

    def prepare_features(self, df: pd.DataFrame, output_dir: str, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for model training or inference."""
        logger.info("Extracting metadata features...")
        metadata_features = self.extract_email_metadata(df)
        
        logger.info("Creating TF-IDF features...")
        tfidf_features = self.create_tfidf_features(df['cleaned_body'], is_training)
        
        # Convert sparse matrix to dense for concatenation
        tfidf_dense = tfidf_features.toarray()
        
        # Combine all features
        X = np.hstack([
            metadata_features.values,
            tfidf_dense
        ])
        
        # Scale features
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Save feature names and parameters if training
        if is_training:
            feature_names = (
                metadata_features.columns.tolist() +
                [f'tfidf_{i}' for i in range(tfidf_dense.shape[1])]
            )
            
            # Save feature names
            feature_names_path = Path(output_dir) / 'feature_names.json'
            with open(feature_names_path, 'w') as f:
                json.dump(feature_names, f)
            
            # Save vectorizer vocabulary
            vocab_path = Path(output_dir) / 'tfidf_vocabulary.json'
            with open(vocab_path, 'w') as f:
                # Convert numpy int64 to regular Python int
                vocabulary = {k: int(v) for k, v in self.tfidf.vocabulary_.items()}
                json.dump(vocabulary, f)
            
            logger.info(f"Saved feature names to {feature_names_path}")
            logger.info(f"Saved TF-IDF vocabulary to {vocab_path}")
        
        return X_scaled

    def prepare_training_data(self, input_file: str, output_dir: str, test_size: float = 0.2) -> None:
        """Prepare data for model training."""
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load processed data
        logger.info(f"Loading data from {input_file}")
        df = pd.read_parquet(input_file)
        
        # Prepare features
        logger.info("Preparing features...")
        X = self.prepare_features(df, output_dir, is_training=True)
        
        # Split into train and test sets
        logger.info(f"Splitting data with test_size={test_size}")
        X_train, X_test = train_test_split(
            X,
            test_size=test_size,
            random_state=self.random_state
        )
        
        # Save train and test sets
        train_path = Path(output_dir) / 'train_features.npz'
        test_path = Path(output_dir) / 'test_features.npz'
        
        np.savez_compressed(train_path, X=X_train)
        np.savez_compressed(test_path, X=X_test)
        
        logger.info(f"Saved training features to {train_path}")
        logger.info(f"Saved test features to {test_path}")
        logger.info(f"Feature matrix shape: {X.shape}")

def main():
    # Initialize feature engineering
    feature_engineer = FeatureEngineer(
        max_features=5000,  # Adjust based on your Mac Mini's capacity
        random_state=42
    )
    
    # Prepare training data
    feature_engineer.prepare_training_data(
        input_file='data/samples/email_processed_1000.parquet',
        output_dir='data/processed',
        test_size=0.2
    )

if __name__ == "__main__":
    main()
