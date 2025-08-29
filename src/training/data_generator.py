"""
Universal synthetic data generator for bias detection training.
Generates diverse datasets with various bias patterns.
"""

import pandas as pd
import numpy as np
from faker import Faker
from typing import Dict, List, Tuple, Any, Optional
import random
import string
from datetime import datetime, timedelta

from config.settings import (
    TRAINING_DATA_DIR, SYNTHETIC_DATASETS_COUNT,
    SAMPLES_PER_DATASET, BIAS_INJECTION_RATES
)
from src.training.pattern_library import pattern_library
from src.utils.logger import get_logger

class DataGenerator:
    """
    Generates synthetic datasets with controlled bias patterns.
    """
    
    def __init__(self):
        self.logger = get_logger('data_generator')
        self.faker = Faker()
        self.pattern_library = pattern_library
        
    def generate_dataset(self, n_samples: int = 1000, 
                        n_features: int = None,
                        bias_patterns: List[str] = None,
                        bias_strength: float = 0.5) -> pd.DataFrame:
        """
        Generate a single synthetic dataset.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features (random if None)
            bias_patterns: List of bias pattern names to apply
            bias_strength: Strength of bias (0-1)
            
        Returns:
            Generated dataset
        """
        if n_features is None:
            n_features = np.random.randint(5, 30)
        
        self.logger.info(f"Generating dataset: {n_samples} samples, {n_features} features")
        
        # Generate base structure
        df = self._generate_base_structure(n_samples, n_features)
        
        # Add protected attributes
        df = self._add_protected_attributes(df)
        
        # Add outcome variables
        df = self._add_outcome_variables(df)
        
        # Apply bias patterns if specified
        if bias_patterns:
            df = self._apply_bias_patterns(df, bias_patterns, bias_strength)
        
        # Add realistic noise and messiness
        df = self._add_data_quality_issues(df)
        
        return df
    
    def _generate_base_structure(self, n_samples: int, n_features: int) -> pd.DataFrame:
        """
        Generate base dataset structure with random features.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            
        Returns:
            Base dataset
        """
        data = {}
        
        for i in range(n_features):
            feature_type = np.random.choice(['numerical', 'categorical', 'binary'])
            col_name = self._generate_random_column_name(i)
            
            if feature_type == 'numerical':
                # Generate various distributions
                dist_type = np.random.choice(['normal', 'uniform', 'exponential', 'bimodal'])
                
                if dist_type == 'normal':
                    data[col_name] = np.random.normal(
                        loc=np.random.uniform(-100, 100),
                        scale=np.random.uniform(1, 50),
                        size=n_samples
                    )
                elif dist_type == 'uniform':
                    low = np.random.uniform(-100, 0)
                    high = np.random.uniform(0, 100)
                    data[col_name] = np.random.uniform(low, high, n_samples)
                elif dist_type == 'exponential':
                    data[col_name] = np.random.exponential(
                        scale=np.random.uniform(1, 10),
                        size=n_samples
                    )
                else:  # bimodal
                    mode1 = np.random.normal(30, 10, n_samples // 2)
                    mode2 = np.random.normal(70, 10, n_samples - n_samples // 2)
                    data[col_name] = np.concatenate([mode1, mode2])
                    np.random.shuffle(data[col_name])
                    
            elif feature_type == 'categorical':
                n_categories = np.random.randint(2, 10)
                categories = [f"cat_{j}" for j in range(n_categories)]
                data[col_name] = np.random.choice(categories, n_samples)
                
            else:  # binary
                data[col_name] = np.random.choice([0, 1], n_samples)
        
        return pd.DataFrame(data)
    
    def _generate_random_column_name(self, index: int) -> str:
        """
        Generate random column name.
        
        Args:
            index: Column index
            
        Returns:
            Random column name
        """
        prefixes = ['feature', 'metric', 'value', 'score', 'attribute', 
                   'variable', 'factor', 'dimension', 'property', 'characteristic']
        suffixes = ['alpha', 'beta', 'gamma', 'delta', 'epsilon',
                   'x', 'y', 'z', 'a', 'b', 'c']
        
        use_prefix = np.random.random() > 0.5
        
        if use_prefix:
            prefix = np.random.choice(prefixes)
            return f"{prefix}_{index}"
        else:
            suffix = np.random.choice(suffixes)
            return f"{suffix}{index}"
    
    def _add_protected_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add protected attribute columns to dataset.
        
        Args:
            df: Base dataset
            
        Returns:
            Dataset with protected attributes
        """
        n_samples = len(df)
        n_protected = np.random.randint(1, 4)  # 1-3 protected attributes
        
        protected_types = ['gender', 'race', 'age', 'nationality', 'religion', 'disability']
        selected_protected = np.random.choice(protected_types, n_protected, replace=False)
        
        for protected_type in selected_protected:
            col_name = self._generate_protected_column_name(protected_type)
            
            if protected_type == 'gender':
                df[col_name] = np.random.choice(
                    ['Male', 'Female', 'Other'],
                    n_samples,
                    p=[0.48, 0.48, 0.04]
                )
            elif protected_type == 'race':
                df[col_name] = np.random.choice(
                    ['White', 'Black', 'Asian', 'Hispanic', 'Other'],
                    n_samples,
                    p=[0.6, 0.13, 0.06, 0.18, 0.03]
                )
            elif protected_type == 'age':
                df[col_name] = np.random.normal(40, 12, n_samples).astype(int)
                df[col_name] = df[col_name].clip(18, 80)
            elif protected_type == 'nationality':
                countries = ['USA', 'UK', 'Canada', 'Australia', 'Other']
                df[col_name] = np.random.choice(countries, n_samples)
            elif protected_type == 'religion':
                religions = ['Christian', 'Muslim', 'Jewish', 'Hindu', 'Buddhist', 'None', 'Other']
                df[col_name] = np.random.choice(religions, n_samples)
            elif protected_type == 'disability':
                df[col_name] = np.random.choice(
                    ['None', 'Physical', 'Cognitive', 'Sensory'],
                    n_samples,
                    p=[0.85, 0.07, 0.05, 0.03]
                )
        
        self.logger.info(f"Added {n_protected} protected attributes")
        return df
    
    def _generate_protected_column_name(self, protected_type: str) -> str:
        """
        Generate column name for protected attribute.
        
        Args:
            protected_type: Type of protected attribute
            
        Returns:
            Column name
        """
        variations = {
            'gender': ['gender', 'sex', 'gender_identity'],
            'race': ['race', 'ethnicity', 'ethnic_group'],
            'age': ['age', 'age_group', 'birth_year'],
            'nationality': ['nationality', 'country_origin', 'citizenship'],
            'religion': ['religion', 'religious_affiliation', 'faith'],
            'disability': ['disability_status', 'disability', 'impairment']
        }
        
        return np.random.choice(variations.get(protected_type, [protected_type]))
    
    def _add_outcome_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add outcome variable columns to dataset.
        
        Args:
            df: Dataset
            
        Returns:
            Dataset with outcome variables
        """
        n_samples = len(df)
        n_outcomes = np.random.randint(1, 3)  # 1-2 outcome variables
        
        outcome_types = ['binary', 'categorical', 'continuous']
        
        for i in range(n_outcomes):
            outcome_type = np.random.choice(outcome_types)
            col_name = self._generate_outcome_column_name()
            
            if outcome_type == 'binary':
                # Initially random, will be modified by bias patterns
                df[col_name] = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
                
            elif outcome_type == 'categorical':
                categories = ['rejected', 'waitlist', 'accepted']
                df[col_name] = np.random.choice(categories, n_samples)
                
            else:  # continuous
                df[col_name] = np.random.normal(50, 15, n_samples)
                df[col_name] = df[col_name].clip(0, 100)
        
        self.logger.info(f"Added {n_outcomes} outcome variables")
        return df
    
    def _generate_outcome_column_name(self) -> str:
        """
        Generate column name for outcome variable.
        
        Returns:
            Column name
        """
        outcomes = [
            'decision', 'outcome', 'result', 'status',
            'approved', 'selected', 'hired', 'admitted',
            'granted', 'qualified', 'passed', 'successful',
            'label', 'target', 'prediction', 'classification'
        ]
        
        return np.random.choice(outcomes)
    
    def _apply_bias_patterns(self, df: pd.DataFrame, pattern_names: List[str],
                            bias_strength: float) -> pd.DataFrame:
        """
        Apply specified bias patterns to dataset.
        
        Args:
            df: Dataset
            pattern_names: List of pattern names
            bias_strength: Strength of bias
            
        Returns:
            Dataset with bias patterns applied
        """
        # Identify protected and outcome columns
        protected_cols = self._identify_protected_columns(df)
        outcome_cols = self._identify_outcome_columns(df)
        
        if not protected_cols or not outcome_cols:
            self.logger.warning("No protected attributes or outcomes found for bias injection")
            return df
        
        df_biased = df.copy()
        
        for pattern_name in pattern_names:
            if pattern_name not in self.pattern_library.patterns:
                continue
            
            pattern = self.pattern_library.patterns[pattern_name]
            self.logger.info(f"Applying bias pattern: {pattern_name}")
            
            # Select columns for pattern application
            protected = np.random.choice(protected_cols)
            outcome = np.random.choice(outcome_cols)
            
            # Apply pattern based on type
            if pattern.bias_type.value == 'direct_discrimination':
                df_biased = pattern.apply_function(
                    df_biased, protected, outcome, 
                    {'disadvantage_rate': 1 - bias_strength,
                     'advantage_rate': bias_strength}
                )
            elif pattern.bias_type.value == 'threshold_shifting':
                # Need a score column
                score_cols = [col for col in df.columns if 'score' in col.lower() or 
                             pd.api.types.is_numeric_dtype(df[col])]
                if score_cols:
                    score = np.random.choice(score_cols)
                    df_biased = pattern.apply_function(
                        df_biased, protected, score, outcome,
                        {'threshold_difference': bias_strength * 0.5}
                    )
            else:
                # Generic pattern application
                try:
                    params = pattern.parameters.copy()
                    # Scale parameters by bias strength
                    for key in params:
                        if isinstance(params[key], (int, float)):
                            params[key] *= bias_strength
                    
                    df_biased = pattern.apply_function(
                        df_biased, protected, outcome, params
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to apply pattern {pattern_name}: {str(e)}")
        
        return df_biased
    
    def _identify_protected_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identify protected attribute columns in dataset.
        
        Args:
            df: Dataset
            
        Returns:
            List of protected column names
        """
        protected = []
        protected_keywords = ['gender', 'sex', 'race', 'ethnic', 'age', 'religion',
                            'disability', 'nationality', 'citizenship']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in protected_keywords):
                protected.append(col)
        
        return protected
    
    def _identify_outcome_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identify outcome variable columns in dataset.
        
        Args:
            df: Dataset
            
        Returns:
            List of outcome column names
        """
        outcomes = []
        outcome_keywords = ['decision', 'outcome', 'result', 'status', 'approved',
                          'selected', 'hired', 'admitted', 'granted', 'qualified',
                          'passed', 'label', 'target', 'prediction']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in outcome_keywords):
                # Check if it's binary or low cardinality
                if df[col].nunique() <= 10:
                    outcomes.append(col)
        
        return outcomes
    
    def _add_data_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add realistic data quality issues.
        
        Args:
            df: Dataset
            
        Returns:
            Dataset with data quality issues
        """
        df_messy = df.copy()
        
        # Missing values (5-20% randomly)
        missing_rate = np.random.uniform(0.05, 0.20)
        for col in df_messy.columns:
            missing_indices = np.random.choice(
                df_messy.index,
                size=int(len(df_messy) * missing_rate),
                replace=False
            )
            df_messy.loc[missing_indices, col] = np.nan
        
        # Outliers in numerical columns (2-5%)
        outlier_rate = np.random.uniform(0.02, 0.05)
        numerical_cols = df_messy.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            outlier_indices = np.random.choice(
                df_messy.index,
                size=int(len(df_messy) * outlier_rate),
                replace=False
            )
            # Add extreme values
            for idx in outlier_indices:
                if np.random.random() > 0.5:
                    df_messy.at[idx, col] = df_messy[col].mean() + 3 * df_messy[col].std()
                else:
                    df_messy.at[idx, col] = df_messy[col].mean() - 3 * df_messy[col].std()
        
        # Duplicates (1-3%)
        duplicate_rate = np.random.uniform(0.01, 0.03)
        n_duplicates = int(len(df_messy) * duplicate_rate)
        
        if n_duplicates > 0:
            duplicate_indices = np.random.choice(df_messy.index, n_duplicates)
            duplicates = df_messy.loc[duplicate_indices]
            df_messy = pd.concat([df_messy, duplicates], ignore_index=True)
        
        # Inconsistent categorical values
        categorical_cols = df_messy.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # Add case variations
            inconsistent_indices = np.random.choice(
                df_messy.index,
                size=int(len(df_messy) * 0.05),
                replace=False
            )
            for idx in inconsistent_indices:
                if pd.notna(df_messy.at[idx, col]):
                    value = str(df_messy.at[idx, col])
                    # Random case variation
                    if np.random.random() > 0.5:
                        df_messy.at[idx, col] = value.upper()
                    else:
                        df_messy.at[idx, col] = value.lower()
        
        return df_messy
    
    def generate_batch(self, n_datasets: int = 100,
                      output_dir: str = None) -> List[str]:
        """
        Generate batch of synthetic datasets.
        
        Args:
            n_datasets: Number of datasets to generate
            output_dir: Output directory
            
        Returns:
            List of generated file paths
        """
        if output_dir is None:
            output_dir = TRAINING_DATA_DIR
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Generating batch of {n_datasets} datasets")
        
        file_paths = []
        
        # Get pattern combinations
        pattern_combinations = self.pattern_library.get_pattern_combinations()
        
        for i in range(n_datasets):
            # Random parameters
            n_samples = np.random.randint(500, 5000)
            n_features = np.random.randint(5, 30)
            
            # Select bias patterns
            if np.random.random() < 0.7:  # 70% have bias
                patterns = random.choice(pattern_combinations)
                bias_strength = np.random.uniform(0.1, 0.9)
            else:  # 30% no bias (control)
                patterns = []
                bias_strength = 0
            
            # Generate dataset
            df = self.generate_dataset(
                n_samples=n_samples,
                n_features=n_features,
                bias_patterns=patterns,
                bias_strength=bias_strength
            )
            
            # Add metadata columns
            df['_bias_injected'] = 1 if patterns else 0
            df['_bias_patterns'] = ','.join(patterns) if patterns else 'none'
            df['_bias_strength'] = bias_strength
            
            # Save dataset
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"synthetic_{i:04d}_{timestamp}.csv"
            filepath = output_dir / filename
            
            df.to_csv(filepath, index=False)
            file_paths.append(str(filepath))
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Generated {i + 1}/{n_datasets} datasets")
        
        self.logger.info(f"Batch generation complete: {n_datasets} datasets saved to {output_dir}")
        return file_paths

# Convenience functions
def generate_synthetic_dataset(n_samples: int = 1000, bias: bool = True) -> pd.DataFrame:
    """
    Generate a single synthetic dataset.
    
    Args:
        n_samples: Number of samples
        bias: Whether to inject bias
        
    Returns:
        Generated dataset
    """
    generator = DataGenerator()
    
    if bias:
        patterns = generator.pattern_library.get_random_patterns(
            np.random.randint(1, 4)
        )
        pattern_names = [p.name for p in patterns]
        bias_strength = np.random.uniform(0.3, 0.8)
    else:
        pattern_names = []
        bias_strength = 0
    
    return generator.generate_dataset(
        n_samples=n_samples,
        bias_patterns=pattern_names,
        bias_strength=bias_strength
    )

def generate_training_data(n_datasets: int = 100) -> List[str]:
    """
    Generate training datasets.
    
    Args:
        n_datasets: Number of datasets
        
    Returns:
        List of file paths
    """
    generator = DataGenerator()
    return generator.generate_batch(n_datasets)

if __name__ == "__main__":
    print("Data Generator Test")
    print("-" * 50)
    
    generator = DataGenerator()
    
    # Generate single dataset with bias
    print("Generating biased dataset...")
    df_biased = generator.generate_dataset(
        n_samples=1000,
        n_features=10,
        bias_patterns=['direct_discrimination', 'threshold_shifting'],
        bias_strength=0.7
    )
    print(f"Generated biased dataset: {df_biased.shape}")
    print(f"Columns: {list(df_biased.columns)[:10]}...")
    
    # Generate dataset without bias
    print("\nGenerating unbiased dataset...")
    df_unbiased = generator.generate_dataset(
        n_samples=1000,
        n_features=10,
        bias_patterns=[],
        bias_strength=0
    )
    print(f"Generated unbiased dataset: {df_unbiased.shape}")
    
    # Generate small batch
    print("\nGenerating batch of 5 datasets...")
    file_paths = generator.generate_batch(n_datasets=5)
    print(f"Generated files:")
    for path in file_paths:
        print(f"  - {path}")
    
    print("\nData generator test completed")