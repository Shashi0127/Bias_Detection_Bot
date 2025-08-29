"""
Model training pipeline for bias detection.
Trains ensemble of models on synthetic data.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from config.settings import (
    MODELS_DIR, TRAINING_DATA_DIR, TEST_SPLIT_RATIO,
    CROSS_VALIDATION_FOLDS, RANDOM_STATE, MODEL_NAME_PREFIX
)
from src.training.data_generator import DataGenerator
from src.utils.logger import get_logger, log_performance

class BiasDetectionTrainer:
    """
    Trains bias detection models using synthetic data.
    """
    
    def __init__(self):
        self.logger = get_logger('trainer')
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.training_history = {}
        
    def prepare_training_data(self, data_dir: Path = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and prepare training data.
        
        Args:
            data_dir: Directory containing training CSV files
            
        Returns:
            Tuple of (features, labels)
        """
        if data_dir is None:
            data_dir = TRAINING_DATA_DIR
        
        self.logger.info(f"Loading training data from {data_dir}")
        
        all_data = []
        csv_files = list(data_dir.glob("*.csv"))
        
        if not csv_files:
            self.logger.warning("No training data found, generating synthetic data...")
            generator = DataGenerator()
            csv_files = generator.generate_batch(n_datasets=100)
            csv_files = [Path(f) for f in csv_files]
        
        for csv_file in csv_files[:100]:  # Limit to 100 files for memory
            try:
                df = pd.read_csv(csv_file)
                all_data.append(df)
            except Exception as e:
                self.logger.warning(f"Failed to load {csv_file}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("No valid training data found")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Loaded {len(combined_df)} samples from {len(all_data)} files")
        
        # Extract features and labels
        X, y = self._extract_features_labels(combined_df)
        
        return X, y
    
    def _extract_features_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from dataset.
        
        Args:
            df: Combined dataset
            
        Returns:
            Tuple of (features, labels)
        """
        # Identify label column (bias flag)
        label_col = None
        for col in df.columns:
            if '_bias_injected' in col or 'bias_flag' in col.lower():
                label_col = col
                break
        
        if label_col is None:
            # Create synthetic labels based on data patterns
            self.logger.warning("No explicit bias labels found, generating from patterns")
            label_col = self._generate_synthetic_labels(df)
        
        # Separate features and labels
        feature_cols = [col for col in df.columns 
                       if not col.startswith('_') and col != label_col]
        
        X = df[feature_cols]
        y = df[label_col]
        
        # Handle categorical features
        X_encoded = self._encode_features(X)
        
        self.feature_names = list(X_encoded.columns)
        
        return X_encoded.values, y.values
    
    def _generate_synthetic_labels(self, df: pd.DataFrame) -> str:
        """
        Generate synthetic bias labels based on data patterns.
        
        Args:
            df: Dataset
            
        Returns:
            Name of generated label column
        """
        # Simple heuristic: detect if there are disparate outcomes
        from src.core.column_analyzer import ColumnAnalyzer
        from src.analysis.statistical_engine import StatisticalEngine
        
        analyzer = ColumnAnalyzer()
        analysis = analyzer.analyze_dataset(df)
        
        protected = [col['name'] for col in analysis['protected_attributes']]
        outcomes = [col['name'] for col in analysis['outcome_variables']]
        
        if protected and outcomes:
            engine = StatisticalEngine()
            bias_results = engine.analyze_bias(df, protected, outcomes)
            
            # Create binary label based on bias detection
            df['_synthetic_bias_label'] = bias_results['bias_detected']
            
            # Add some noise for variety
            noise_indices = np.random.choice(
                df.index,
                size=int(len(df) * 0.1),
                replace=False
            )
            df.loc[noise_indices, '_synthetic_bias_label'] = ~df.loc[noise_indices, '_synthetic_bias_label']
        else:
            # Random labels if no clear patterns
            df['_synthetic_bias_label'] = np.random.choice([0, 1], len(df))
        
        return '_synthetic_bias_label'
    
    def _encode_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Encoded features
        """
        X_encoded = X.copy()
        
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object':
                # One-hot encode categorical features
                dummies = pd.get_dummies(X_encoded[col], prefix=col, drop_first=True)
                X_encoded = pd.concat([X_encoded.drop(col, axis=1), dummies], axis=1)
        
        # Fill any missing values
        X_encoded = X_encoded.fillna(X_encoded.mean())
        
        return X_encoded
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train ensemble of bias detection models.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Training results
        """
        self.logger.info("Starting model training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_STATE, stratify=y
        )
        
        self.logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        results = {}
        
        # Train Gradient Boosting
        self.logger.info("Training Gradient Boosting Classifier...")
        gb_model = self._train_gradient_boosting(X_train_scaled, y_train)
        results['gradient_boosting'] = self._evaluate_model(
            gb_model, X_test_scaled, y_test, 'Gradient Boosting'
        )
        self.models['gradient_boosting'] = gb_model
        
        # Train Random Forest
        self.logger.info("Training Random Forest Classifier...")
        rf_model = self._train_random_forest(X_train_scaled, y_train)
        results['random_forest'] = self._evaluate_model(
            rf_model, X_test_scaled, y_test, 'Random Forest'
        )
        self.models['random_forest'] = rf_model
        
        # Train Logistic Regression
        self.logger.info("Training Logistic Regression...")
        lr_model = self._train_logistic_regression(X_train_scaled, y_train)
        results['logistic_regression'] = self._evaluate_model(
            lr_model, X_test_scaled, y_test, 'Logistic Regression'
        )
        self.models['logistic_regression'] = lr_model
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
        self.logger.info(f"Best model: {best_model_name} (F1: {results[best_model_name]['f1_score']:.4f})")
        
        return results
    
    def _train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray):
       """
       Train Gradient Boosting classifier.
       
       Args:
           X_train: Training features
           y_train: Training labels
           
       Returns:
           Trained model
       """
       params = {
           'n_estimators': [50, 100, 150],
           'learning_rate': [0.05, 0.1, 0.15],
           'max_depth': [3, 5, 7],
           'min_samples_split': [10, 20],
           'min_samples_leaf': [5, 10]
       }
       
       gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
       
       # Grid search with cross-validation
       grid_search = GridSearchCV(
           gb, params, cv=3, scoring='f1', n_jobs=-1, verbose=0
       )
       grid_search.fit(X_train, y_train)
       
       self.logger.info(f"Best GB params: {grid_search.best_params_}")
       return grid_search.best_estimator_
   
    def _train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray):
       """
       Train Random Forest classifier.
       
       Args:
           X_train: Training features
           y_train: Training labels
           
       Returns:
           Trained model
       """
       params = {
           'n_estimators': [50, 100, 150],
           'max_depth': [10, 20, None],
           'min_samples_split': [5, 10],
           'min_samples_leaf': [2, 4]
       }
       
       rf = RandomForestClassifier(random_state=RANDOM_STATE)
       
       grid_search = GridSearchCV(
           rf, params, cv=3, scoring='f1', n_jobs=-1, verbose=0
       )
       grid_search.fit(X_train, y_train)
       
       self.logger.info(f"Best RF params: {grid_search.best_params_}")
       return grid_search.best_estimator_
   
    def _train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray):
       """
       Train Logistic Regression classifier.
       
       Args:
           X_train: Training features
           y_train: Training labels
           
       Returns:
           Trained model
       """
       params = {
           'C': [0.01, 0.1, 1, 10],
           'penalty': ['l2'],
           'solver': ['lbfgs', 'liblinear']
       }
       
       lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
       
       grid_search = GridSearchCV(
           lr, params, cv=3, scoring='f1', n_jobs=-1, verbose=0
       )
       grid_search.fit(X_train, y_train)
       
       self.logger.info(f"Best LR params: {grid_search.best_params_}")
       return grid_search.best_estimator_
   
    def _evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str) -> Dict[str, Any]:
       """
       Evaluate model performance.
       
       Args:
           model: Trained model
           X_test: Test features
           y_test: Test labels
           model_name: Name of the model
           
       Returns:
           Evaluation metrics
       """
       y_pred = model.predict(X_test)
       y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
       
       metrics = {
           'accuracy': accuracy_score(y_test, y_pred),
           'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
           'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
           'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
           'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) == 2 else 0,
           'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
           'classification_report': classification_report(y_test, y_pred, output_dict=True)
       }
       
       # Cross-validation score
       cv_scores = cross_val_score(model, X_test, y_test, cv=CROSS_VALIDATION_FOLDS, scoring='f1')
       metrics['cv_mean'] = cv_scores.mean()
       metrics['cv_std'] = cv_scores.std()
       
       self.logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                       f"F1: {metrics['f1_score']:.4f}, AUC: {metrics['roc_auc']:.4f}")
       
       return metrics
   
    def save_models(self, model_dir: Path = None) -> Dict[str, str]:
       """
       Save trained models and metadata.
       
       Args:
           model_dir: Directory to save models
           
       Returns:
           Dictionary of saved file paths
       """
       if model_dir is None:
           model_dir = MODELS_DIR
       
       model_dir.mkdir(parents=True, exist_ok=True)
       
       timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
       saved_files = {}
       
       # Save models
       for model_name, model in self.models.items():
           filename = f"{MODEL_NAME_PREFIX}_{model_name}_{timestamp}.pkl"
           filepath = model_dir / filename
           joblib.dump(model, filepath)
           saved_files[f"{model_name}_model"] = str(filepath)
           self.logger.info(f"Saved {model_name} model to {filepath}")
       
       # Save scalers
       for scaler_name, scaler in self.scalers.items():
           filename = f"{MODEL_NAME_PREFIX}_{scaler_name}_scaler_{timestamp}.pkl"
           filepath = model_dir / filename
           joblib.dump(scaler, filepath)
           saved_files[f"{scaler_name}_scaler"] = str(filepath)
           self.logger.info(f"Saved {scaler_name} scaler to {filepath}")
       
       # Save feature names
       features_file = model_dir / f"{MODEL_NAME_PREFIX}_features_{timestamp}.json"
       with open(features_file, 'w') as f:
           json.dump(self.feature_names, f, indent=2)
       saved_files['features'] = str(features_file)
       
       # Save training metadata
       metadata = {
           'timestamp': timestamp,
           'feature_count': len(self.feature_names),
           'models': list(self.models.keys()),
           'training_history': self.training_history
       }
       
       metadata_file = model_dir / f"{MODEL_NAME_PREFIX}_metadata_{timestamp}.json"
       with open(metadata_file, 'w') as f:
           json.dump(metadata, f, indent=2)
       saved_files['metadata'] = str(metadata_file)
       
       self.logger.info(f"All models and metadata saved to {model_dir}")
       return saved_files
   
    def load_models(self, model_dir: Path = None) -> bool:
       """
       Load saved models.
       
       Args:
           model_dir: Directory containing models
           
       Returns:
           True if successful
       """
       if model_dir is None:
           model_dir = MODELS_DIR
       
       try:
           # Find latest model files
           model_files = list(model_dir.glob(f"{MODEL_NAME_PREFIX}_*.pkl"))
           
           if not model_files:
               self.logger.warning("No model files found")
               return False
           
           # Group by timestamp
           timestamps = {}
           for file in model_files:
               parts = file.stem.split('_')
               if len(parts) >= 4:
                   timestamp = f"{parts[-2]}_{parts[-1]}"
                   if timestamp not in timestamps:
                       timestamps[timestamp] = []
                   timestamps[timestamp].append(file)
           
           # Use latest timestamp
           latest_timestamp = max(timestamps.keys())
           latest_files = timestamps[latest_timestamp]
           
           # Load models
           for file in latest_files:
               if 'gradient_boosting' in file.stem:
                   self.models['gradient_boosting'] = joblib.load(file)
                   self.logger.info(f"Loaded gradient boosting model from {file}")
               elif 'random_forest' in file.stem:
                   self.models['random_forest'] = joblib.load(file)
                   self.logger.info(f"Loaded random forest model from {file}")
               elif 'logistic_regression' in file.stem:
                   self.models['logistic_regression'] = joblib.load(file)
                   self.logger.info(f"Loaded logistic regression model from {file}")
               elif 'scaler' in file.stem:
                   self.scalers['standard'] = joblib.load(file)
                   self.logger.info(f"Loaded scaler from {file}")
           
           # Load feature names
           feature_files = list(model_dir.glob(f"{MODEL_NAME_PREFIX}_features_{latest_timestamp}.json"))
           if feature_files:
               with open(feature_files[0], 'r') as f:
                   self.feature_names = json.load(f)
               self.logger.info(f"Loaded {len(self.feature_names)} feature names")
           
           return True
           
       except Exception as e:
           self.logger.error(f"Failed to load models: {str(e)}")
           return False
   
    def run_training_pipeline(self, n_datasets: int = 100) -> Dict[str, Any]:
       """
       Run complete training pipeline.
       
       Args:
           n_datasets: Number of datasets to generate
           
       Returns:
           Training results
       """
       self.logger.info("Starting complete training pipeline")
       start_time = datetime.now()
       
       # Generate training data if needed
       csv_files = list(TRAINING_DATA_DIR.glob("*.csv"))
       if len(csv_files) < n_datasets:
           self.logger.info(f"Generating {n_datasets} synthetic datasets...")
           generator = DataGenerator()
           generator.generate_batch(n_datasets)
       
       # Prepare data
       X, y = self.prepare_training_data()
       
       # Train models
       results = self.train_models(X, y)
       
       # Save models
       saved_files = self.save_models()
       
       # Calculate training time
       training_time = (datetime.now() - start_time).total_seconds()
       
       # Log performance
       log_performance('training_pipeline', training_time, {
           'n_samples': len(X),
           'n_features': X.shape[1] if len(X.shape) > 1 else 1,
           'models_trained': len(self.models)
       })
       
       self.logger.info(f"Training pipeline completed in {training_time:.2f} seconds")
       
       return {
           'results': results,
           'saved_files': saved_files,
           'training_time': training_time
       }

# Convenience functions
def train_bias_detection_models(n_datasets: int = 100) -> Dict[str, Any]:
   """
   Train bias detection models.
   
   Args:
       n_datasets: Number of training datasets
       
   Returns:
       Training results
   """
   trainer = BiasDetectionTrainer()
   return trainer.run_training_pipeline(n_datasets)

def load_trained_models() -> BiasDetectionTrainer:
   """
   Load pre-trained models.
   
   Returns:
       Trainer instance with loaded models
   """
   trainer = BiasDetectionTrainer()
   success = trainer.load_models()
   if not success:
       raise ValueError("Failed to load models")
   return trainer

if __name__ == "__main__":
   import argparse
   
   parser = argparse.ArgumentParser(description="Bias Detection Model Training")
   parser.add_argument('command', choices=['train', 'generate', 'evaluate'],
                      help='Command to execute')
   parser.add_argument('--datasets', type=int, default=100,
                      help='Number of datasets to generate/use')
   parser.add_argument('--samples', type=int, default=1000,
                      help='Samples per dataset')
   
   args = parser.parse_args()
   
   print("Bias Detection Training Pipeline")
   print("-" * 50)
   
   if args.command == 'generate':
       print(f"Generating {args.datasets} synthetic datasets...")
       generator = DataGenerator()
       file_paths = generator.generate_batch(args.datasets)
       print(f"Generated {len(file_paths)} datasets")
       
   elif args.command == 'train':
       print(f"Training models with {args.datasets} datasets...")
       trainer = BiasDetectionTrainer()
       results = trainer.run_training_pipeline(args.datasets)
       
       print("\nTraining Results:")
       for model_name, metrics in results['results'].items():
           print(f"\n{model_name}:")
           print(f"  Accuracy: {metrics['accuracy']:.4f}")
           print(f"  F1 Score: {metrics['f1_score']:.4f}")
           print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
       
       print(f"\nModels saved to: {MODELS_DIR}")
       print(f"Training time: {results['training_time']:.2f} seconds")
       
   elif args.command == 'evaluate':
       print("Loading and evaluating saved models...")
       trainer = BiasDetectionTrainer()
       if trainer.load_models():
           print(f"Loaded {len(trainer.models)} models")
           print(f"Models: {list(trainer.models.keys())}")
           print(f"Features: {len(trainer.feature_names)}")
       else:
           print("No models found. Run training first.")
   
   print("\nTraining pipeline completed")
