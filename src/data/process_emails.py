import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
import re
from pathlib import Path
import logging
from typing import Generator, List, Dict
import email
from email.parser import Parser
from email.policy import default

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailProcessor:
    def __init__(self, batch_size: int = 1000):
        """Initialize the email processor with memory-efficient settings."""
        self.batch_size = batch_size
        self.email_parser = Parser(policy=default)
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """Clean text content efficiently."""
        # Remove email forwarding markers
        text = re.sub(r'(-+)?\s*Forwarded by.*?(-+)?', '', text, flags=re.DOTALL)
        # Remove reply markers
        text = re.sub(r'On.*wrote:.*', '', text, flags=re.DOTALL)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.IGNORECASE)
        # Remove multiple newlines and spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def tokenize_and_clean(self, text: str) -> List[str]:
        """Tokenize and clean text efficiently."""
        # Tokenize
        tokens = word_tokenize(text.lower())
        # Remove stopwords and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        return tokens

    def extract_email_features(self, email_content: str) -> Dict:
        """Extract basic features from email content."""
        try:
            msg = self.email_parser.parsestr(email_content)
            
            # Basic features
            features = {
                'subject': msg.get('subject', ''),
                'from': msg.get('from', ''),
                'to': msg.get('to', ''),
                'date': msg.get('date', ''),
                'body': self.get_email_body(msg),
            }
            
            # Clean the body text
            features['cleaned_body'] = self.clean_text(features['body'])
            
            # Add tokenized text
            features['tokens'] = self.tokenize_and_clean(features['cleaned_body'])
            
            return features
        except Exception as e:
            logger.error(f"Error processing email: {str(e)}")
            return None

    def get_email_body(self, msg: email.message.Message) -> str:
        """Extract email body efficiently."""
        body = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body.append(part.get_payload(decode=True).decode('utf-8', errors='ignore'))
        else:
            body.append(msg.get_payload(decode=True).decode('utf-8', errors='ignore'))
        return '\n'.join(body)

    def process_file_batch(self, file_paths: List[str]) -> pd.DataFrame:
        """Process a batch of email files."""
        features_list = []
        for file_path in tqdm(file_paths, desc="Processing files"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                features = self.extract_email_features(content)
                if features:
                    features['file_path'] = file_path
                    features_list.append(features)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        return pd.DataFrame(features_list)

    def create_sample_dataset(self, input_dir: str, output_dir: str, sample_size: int = 1000):
        """Create a smaller sample dataset for testing."""
        all_files = list(Path(input_dir).rglob('*.txt'))
        if len(all_files) > sample_size:
            sample_files = pd.Series(all_files).sample(n=sample_size).tolist()
        else:
            sample_files = all_files
        
        df = self.process_file_batch(sample_files)
        
        # Save as parquet for efficiency
        output_path = Path(output_dir) / 'email_sample.parquet'
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved sample dataset to {output_path}")
        return df

def main():
    # Initialize processor
    processor = EmailProcessor()
    
    # Create sample dataset
    sample_df = processor.create_sample_dataset(
        input_dir='data/raw',
        output_dir='data/samples',
        sample_size=1000  # Adjust based on your Mac Mini's capacity
    )
    
    logger.info(f"Processed {len(sample_df)} emails successfully")

if __name__ == "__main__":
    main()
