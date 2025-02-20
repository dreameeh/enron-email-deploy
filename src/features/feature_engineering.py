import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import logging
from typing import Tuple, Dict
import re
from datetime import datetime
import pyarrow as pa

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, max_features: int = 10000):
        """Initialize feature engineering with memory-efficient settings."""
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.label_encoders = {}

    def extract_time_features(self, date_str: str) -> Dict:
        """Extract time-based features from email date."""
        try:
            # Handle various date formats
            date_formats = [
                '%a, %d %b %Y %H:%M:%S %z',
                '%d %b %Y %H:%M:%S %z',
                '%a, %d %b %Y %H:%M:%S',
            ]
            
            date_obj = None
            for fmt in date_formats:
                try:
                    date_obj = datetime.strptime(date_str.strip(), fmt)
                    break
                except ValueError:
                    continue
            
            if date_obj is None:
                return {'hour': -1, 'day_of_week': -1, 'month': -1}
            
            return {
                'hour': date_obj.hour,
                'day_of_week': date_obj.weekday(),
                'month': date_obj.month
            }
        except Exception as e:
            logger.error(f"Error extracting time features: {str(e)}")
            return {'hour': -1, 'day_of_week': -1, 'month': -1}

    def compute_text_stats(self, text: str) -> Dict:
        """Compute basic text statistics."""
        return {
            'text_length': len(text),
            'word_count': len(text.split()),
            'avg_word_length': np.mean([len(w) for w in text.split()]) if text else 0,
            'contains_question': '?' in text,
            'contains_exclamation': '!' in text,
            'url_count': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        }

    def create_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Create features from email data efficiently."""
        logger.info("Starting feature engineering...")
        
        # Create basic text features
        text_features = df['cleaned_body'].apply(self.compute_text_stats)
        text_features_df = pd.DataFrame.from_records(text_features)
        
        # Create time features
        time_features = df['date'].apply(self.extract_time_features)
        time_features_df = pd.DataFrame.from_records(time_features)
        
        # TF-IDF features (memory efficient)
        if is_training:
            tfidf_features = self.tfidf.fit_transform(df['cleaned_body'])
        else:
            tfidf_features = self.tfidf.transform(df['cleaned_body'])
        
        # Convert sparse matrix to DataFrame efficiently
        tfidf_df = pd.DataFrame.sparse.from_spmatrix(
            tfidf_features,
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        # Combine all features
        feature_df = pd.concat([
            df[['from', 'to']],  # Keep original columns needed for analysis
            text_features_df,
            time_features_df,
            tfidf_df
        ], axis=1)
        
        # Convert categorical variables
        for col in ['from', 'to']:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                feature_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(feature_df[col])
            else:
                # Handle unknown categories in test data
                unknown_categories = ~feature_df[col].isin(self.label_encoders[col].classes_)
                feature_df.loc[unknown_categories, col] = 'UNKNOWN'
                feature_df[f'{col}_encoded'] = self.label_encoders[col].transform(feature_df[col])
        
        logger.info(f"Created {feature_df.shape[1]} features")
        return feature_df

    def save_features(self, df: pd.DataFrame, output_path: str):
        """Save features efficiently using parquet format."""
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved features to {output_path}")

def main():
    # Load sample data
    df = pd.read_parquet('data/samples/email_sample.parquet')
    
    # Initialize feature engineering
    feature_engineer = FeatureEngineer(max_features=5000)  # Adjust based on your Mac Mini's capacity
    
    # Create and save features
    features_df = feature_engineer.create_features(df)
    feature_engineer.save_features(features_df, 'data/processed/email_features.parquet')

if __name__ == "__main__":
    main()
