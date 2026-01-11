from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
import os
import json
import logging
from typing import Optional, Tuple, Dict
import shutil
from datetime import datetime
from peft import PeftModel, LoraConfig

logger = logging.getLogger(__name__)

class ModelCache:
    def __init__(self, cache_dir: str = "models/cache"):
        """Initialize model cache manager."""
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.metadata_file = os.path.join(self.cache_dir, "cache_metadata.json")
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def save_fine_tuned_model(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        version: str,
        training_args: Dict = None
    ):
        """Save fine-tuned model with LoRA adapters."""
        # Create version directory
        model_dir = os.path.join(self.cache_dir, f"enron_llama_{version}")
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Saving fine-tuned model to {model_dir}")
        
        # Save LoRA adapter weights
        model.save_pretrained(model_dir)
        
        # Save tokenizer
        tokenizer.save_pretrained(model_dir)
        
        # Save training arguments
        if training_args:
            with open(os.path.join(model_dir, "training_args.json"), 'w') as f:
                json.dump(training_args, f, indent=2)
        
        # Update metadata
        self.metadata[version] = {
            'path': model_dir,
            'timestamp': datetime.now().isoformat(),
            'training_args': training_args or {},
            'type': 'lora_adapter'
        }
        self._save_metadata()

    def load_fine_tuned_model(
        self,
        base_model_name: str,
        version: str = "latest"
    ) -> Tuple[Optional[PeftModel], Optional[PreTrainedTokenizer]]:
        """Load fine-tuned model with LoRA adapters."""
        # Get latest version if not specified
        if version == "latest" and self.metadata:
            version = max(self.metadata.keys())
        
        if version not in self.metadata:
            logger.warning(f"No cached model found for version {version}")
            return None, None
        
        model_dir = self.metadata[version]['path']
        if not os.path.exists(model_dir):
            logger.warning(f"Cache path {model_dir} does not exist")
            return None, None
        
        logger.info(f"Loading fine-tuned model from {model_dir}")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            load_in_8bit=True
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(
            base_model,
            model_dir,
            torch_dtype=torch.float16,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        return model, tokenizer

    def get_training_args(self, version: str) -> Optional[Dict]:
        """Get training arguments for a specific version."""
        if version in self.metadata:
            return self.metadata[version].get('training_args')
        return None

    def list_versions(self) -> Dict:
        """List all cached model versions."""
        return {
            version: {
                'timestamp': info['timestamp'],
                'type': info['type']
            }
            for version, info in self.metadata.items()
        }

    def cleanup(self, keep_versions: int = 3):
        """Clean up old model versions, keeping only the N most recent."""
        if len(self.metadata) <= keep_versions:
            return
        
        # Sort versions by timestamp
        sorted_versions = sorted(
            self.metadata.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        # Remove old versions
        for version, info in sorted_versions[keep_versions:]:
            path = info['path']
            if os.path.exists(path):
                shutil.rmtree(path)
            del self.metadata[version]
        
        self._save_metadata()

    def get_model_path(self, version: str) -> str:
        """Get the path to a specific model version."""
        if not version:
            raise ValueError("Version cannot be empty")
            
        version_path = os.path.join(self.cache_dir, f"enron_llama_{version}")
        if not os.path.exists(version_path):
            raise ValueError(f"Model version {version} not found in {version_path}")
            
        return version_path