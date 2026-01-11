import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from pathlib import Path
import json
from typing import Dict, Any, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        random_state: int = 42
    ):
        """Initialize the email classifier."""
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        
    def load_data(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load training and test data."""
        train_data = np.load(Path(data_dir) / 'train_features.npz')
        test_data = np.load(Path(data_dir) / 'test_features.npz')
        
        X_train = train_data['X']
        X_test = test_data['X']
        
        # For now, we'll create dummy labels for demonstration
        # In a real scenario, these would come from your labeled dataset
        y_train = np.random.randint(0, 2, size=X_train.shape[0])
        y_test = np.random.randint(0, 2, size=X_test.shape[0])
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model."""
        logger.info("Training model...")
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model and return metrics."""
        logger.info("Evaluating model...")
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        return metrics
    
    def save_model(self, output_dir: str, metrics: Dict[str, Any]) -> None:
        """Save the trained model and metrics."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / 'email_classifier.joblib'
        joblib.dump(self.model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save metrics
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save feature importances
        importances = pd.DataFrame({
            'feature': range(self.model.n_features_in_),
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importances_path = output_dir / 'feature_importances.csv'
        importances.to_csv(importances_path, index=False)
        logger.info(f"Saved feature importances to {importances_path}")

def main():
    # Initialize classifier
    classifier = EmailClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    
    # Load data
    X_train, X_test, y_train, y_test = classifier.load_data('data/processed')
    
    # Train model
    classifier.train(X_train, y_train)
    
    # Evaluate model
    metrics = classifier.evaluate(X_test, y_test)
    
    # Save model and metrics
    classifier.save_model('models', metrics)

if __name__ == "__main__":
    main()
