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
        
        # Download required NLTK data
        nltk_resources = ['punkt', 'stopwords', 'punkt_tab']
        for resource in nltk_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource)
        
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """Clean text content efficiently."""
        if not isinstance(text, str):
            return ""
            
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
        if not isinstance(text, str):
            return []
            
        # Tokenize
        tokens = word_tokenize(text.lower())
        # Remove stopwords and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        return tokens

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of emails from DataFrame."""
        logger.info(f"Processing batch of {len(df)} emails")
        
        # Clean body text
        df['cleaned_body'] = df['message'].apply(self.clean_text)
        
        # Add tokenized text
        df['tokens'] = df['cleaned_body'].apply(self.tokenize_and_clean)
        
        # Add token count
        df['token_count'] = df['tokens'].apply(len)
        
        return df

    def process_csv(self, input_file: str, output_dir: str, sample_size: int = None):
        """Process emails from CSV file in batches."""
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Read CSV in chunks
        chunk_iterator = pd.read_csv(
            input_file,
            chunksize=self.batch_size,
            encoding='utf-8',
            on_bad_lines='skip'  # Skip problematic lines
        )
        
        processed_chunks = []
        total_processed = 0
        
        for chunk in tqdm(chunk_iterator, desc="Processing email chunks"):
            if sample_size and total_processed >= sample_size:
                break
                
            # Process the chunk
            processed_chunk = self.process_dataframe(chunk)
            processed_chunks.append(processed_chunk)
            
            total_processed += len(chunk)
            
            # If we have enough chunks or it's the last chunk, save to parquet
            if len(processed_chunks) * self.batch_size >= 10000 or (sample_size and total_processed >= sample_size):
                # Combine chunks
                combined_df = pd.concat(processed_chunks, ignore_index=True)
                
                # If sample_size is specified, take only what we need
                if sample_size:
                    combined_df = combined_df.head(sample_size)
                
                # Save to parquet
                output_path = Path(output_dir) / f'email_processed_{total_processed}.parquet'
                combined_df.to_parquet(output_path, index=False)
                logger.info(f"Saved processed data to {output_path}")
                
                # Clear processed chunks to free memory
                processed_chunks = []
        
        # Save any remaining chunks
        if processed_chunks:
            combined_df = pd.concat(processed_chunks, ignore_index=True)
            if sample_size:
                combined_df = combined_df.head(sample_size)
            output_path = Path(output_dir) / f'email_processed_final.parquet'
            combined_df.to_parquet(output_path, index=False)
            logger.info(f"Saved final batch to {output_path}")
        
        logger.info(f"Processed {total_processed} emails successfully")

def main():
    # Initialize processor
    processor = EmailProcessor(batch_size=1000)
    
    # Process sample dataset
    processor.process_csv(
        input_file='data/raw/emails.csv',
        output_dir='data/samples',
        sample_size=1000  # Adjust based on your Mac Mini's capacity
    )

if __name__ == "__main__":
    main()
