import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftConfig,
    PeftModel
)
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
import logging
import json
from datetime import datetime
import os
from pathlib import Path
import sys
import random
from typing import List

sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.models.model_cache import ModelCache

class LlamaEmailTuner:
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        learning_rate: float = 3e-4,
        num_epochs: int = 1,
        batch_size: int = 8,
        pre_prompts: List[str] = None,
        max_length: int = 256,
        lora_r: int = 4,
        lora_alpha: int = 16
    ):
        """Initialize the LlamaEmailTuner.
        
        Args:
            model_name: Name of the base model to fine-tune
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            pre_prompts: List of pre-prompts to use during training (max 5)
            max_length: Maximum sequence length
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
        """
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.pre_prompts = pre_prompts[:5] if pre_prompts else []
        self.max_length = max_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        
        # Initialize cache
        self.cache = ModelCache()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model in fp16
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": "cpu"}  # Force CPU usage since GPU is not available
        )
        
        # Prepare model for LoRA training
        self.model = prepare_model_for_kbit_training(self.base_model)
        
        # Configure LoRA
        self.lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Create LoRA model
        self.model = get_peft_model(self.base_model, self.lora_config)
    
    def prepare_training_data(self, emails_df: pd.DataFrame) -> Dataset:
        """Prepare email data for training with optional pre-prompts."""
        logger.info("Preparing training data...")
        
        def format_conversation(row):
            # Get email content
            content = row['message']
            
            # If we have pre-prompts, randomly select one
            if self.pre_prompts:
                prompt = random.choice(self.pre_prompts)
                formatted = f"{prompt}\n\nYou are Phillip Allen from Enron. Write an email response:\n\n{content}</s>"
            else:
                formatted = f"You are Phillip Allen from Enron. Write an email response:\n\n{content}</s>"
            
            return formatted
        
        # Format conversations
        conversations = emails_df['message'].apply(format_conversation).tolist()
        
        # Tokenize
        tokenized = self.tokenizer(
            conversations,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create dataset
        dataset = Dataset.from_dict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids'].clone()
        })
        
        return dataset
    
    def train(self, dataset: Dataset, output_dir: str = "models/checkpoints"):
        """Train the model using LoRA."""
        logger.info("Starting training...")
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=8,
            learning_rate=self.learning_rate,
            fp16=False,
            logging_steps=1,
            save_strategy="epoch",
            warmup_ratio=0.1,
            optim="adamw_torch",
            remove_unused_columns=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Train the model
        trainer.train()
        
        # Save model
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_path = os.path.join("models/cache", version)
        os.makedirs(version_path, exist_ok=True)
        
        # Save LoRA config and model
        self.model.save_pretrained(version_path)
        self.tokenizer.save_pretrained(version_path)
        
        # Save training args and metadata
        metadata = {
            'training_args': training_args.to_dict(),
            'pre_prompts': self.pre_prompts
        }
        
        self.cache.save_fine_tuned_model(
            model=self.model,
            tokenizer=self.tokenizer,
            version=version,
            training_args=metadata
        )
        
        logger.info(f"Model saved as version: {version}")
        return version

    def load_model(self, model_path: str):
        """Load a saved model for inference."""
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Load base model with PeftConfig to handle LoRA
            config = PeftConfig.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.float16,
                device_map={"": "cpu"},  # Force CPU usage since GPU is not available
                trust_remote_code=True
            )
            
            # Load tokenizer from base model
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.base_model_name_or_path,
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load and apply LoRA adapter
            logger.info("Loading LoRA adapter...")
            model = PeftModel.from_pretrained(
                model,
                model_path,
                torch_dtype=torch.float16,
                device_map={"": "cpu"}
            )
            
            # Merge LoRA weights with base model for inference
            logger.info("Merging LoRA weights with base model...")
            model = model.merge_and_unload()
            
            # Set to evaluation mode
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise Exception(f"Failed to load model: {str(e)}")

def main():
    # Initialize tuner
    tuner = LlamaEmailTuner()
    
    # Load data from samples directory
    df = pd.read_parquet('data/samples/email_processed_1000.parquet')
    
    # Prepare dataset
    dataset = tuner.prepare_training_data(df)
    
    # Train
    version = tuner.train(dataset)
    logger.info("Training complete!")
    logger.info(f"Model saved as version: {version}")
    
    # Show available versions
    versions = tuner.cache.list_versions()
    logger.info(f"Available versions: {versions}")

if __name__ == "__main__":
    main()
