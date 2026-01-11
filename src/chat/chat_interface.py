import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List
import re

class EnronChatbot:
    def __init__(self, data_dir: str = 'data/samples'):
        """Initialize the Enron chatbot."""
        self.data_dir = Path(data_dir)
        
        # Load email data
        self.emails_df = pd.read_parquet(self.data_dir / 'email_processed_1000.parquet')
        
        # Initialize conversation history
        self.conversation_history: List[Dict] = []

    def clean_text(self, text: str) -> str:
        """Clean and preprocess text data."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    def get_relevant_response(self, user_input: str) -> str:
        """Get a relevant response based on the user's input."""
        # Clean user input
        cleaned_input = self.clean_text(user_input)
        
        # Get similar emails based on text similarity
        similarities = []
        for _, row in self.emails_df.iterrows():
            similarity = self._compute_similarity(cleaned_input, self.clean_text(row['message']))
            similarities.append(similarity)
        
        # Get the most similar email
        most_similar_idx = np.argmax(similarities)
        response = self._format_response(self.emails_df.iloc[most_similar_idx])
        
        # Update conversation history
        self.conversation_history.append({
            'user': user_input,
            'bot': response,
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        # Simple similarity based on common words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0
    
    def _format_response(self, email_row: pd.Series) -> str:
        """Format the email as a chat response."""
        # Clean and format the message
        message = email_row['message']
        
        # Remove email headers if present
        if '\n\n' in message:
            message = message.split('\n\n', 1)[1]
        
        # Clean up newlines and spaces
        message = ' '.join(message.split())
        
        # Truncate very long messages
        if len(message) > 200:
            message = message[:197] + "..."
        
        return message

    def start_chat(self):
        """Start an interactive chat session."""
        print("Welcome! You're now chatting with Phillip Allen from Enron.")
        print("Type 'quit' to end the conversation.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nGoodbye! Thanks for chatting.")
                    break
                
                response = self.get_relevant_response(user_input)
                print(f"\nPhillip Allen: {response}\n")
            except Exception as e:
                print(f"\nSorry, I encountered an error: {str(e)}\n")

def main():
    chatbot = EnronChatbot()
    chatbot.start_chat()

if __name__ == "__main__":
    main()
