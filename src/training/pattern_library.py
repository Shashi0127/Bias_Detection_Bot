"""
Bias pattern library containing templates for synthetic data generation.
Defines various types of bias patterns that can occur in real-world data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import get_logger

class BiasType(Enum):
    """Enumeration of bias pattern types."""
    DIRECT = "direct_discrimination"
    THRESHOLD = "threshold_shifting"
    PROXY = "proxy_discrimination"
    INTERSECTIONAL = "intersectional"
    GLASS_CEILING = "glass_ceiling"
    STATISTICAL = "statistical_discrimination"
    HISTORICAL = "historical_bias"
    SAMPLING = "sampling_bias"
    REPRESENTATION = "representation_bias"
    AGGREGATION = "aggregation_bias"

@dataclass
class BiasPattern:
    """Data class for bias pattern definition."""
    name: str
    bias_type: BiasType
    description: str
    severity: float  # 0-1 scale
    apply_function: Callable
    parameters: Dict[str, Any]

class PatternLibrary:
    """
    Library of bias patterns for synthetic data generation.
    """
    
    def __init__(self):
        self.logger = get_logger('pattern_library')
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, BiasPattern]:
        """
        Initialize all bias pattern templates.
        
        Returns:
            Dictionary of bias patterns
        """
        patterns = {}
        
        # Direct discrimination pattern
        patterns['direct_discrimination'] = BiasPattern(
            name='direct_discrimination',
            bias_type=BiasType.DIRECT,
            description='Direct discrimination based on protected attribute',
            severity=0.8,
            apply_function=self._apply_direct_discrimination,
            parameters={'disadvantage_rate': 0.7, 'advantage_rate': 0.3}
        )
        
        # Threshold shifting pattern
        patterns['threshold_shifting'] = BiasPattern(
            name='threshold_shifting',
            bias_type=BiasType.THRESHOLD,
            description='Different thresholds for different groups',
            severity=0.6,
            apply_function=self._apply_threshold_shifting,
            parameters={'threshold_difference': 0.2}
        )
        
        # Proxy discrimination pattern
        patterns['proxy_discrimination'] = BiasPattern(
            name='proxy_discrimination',
            bias_type=BiasType.PROXY,
            description='Discrimination through correlated features',
            severity=0.5,
            apply_function=self._apply_proxy_discrimination,
            parameters={'correlation_strength': 0.7}
        )
        
        # Intersectional bias pattern
        patterns['intersectional_bias'] = BiasPattern(
            name='intersectional_bias',
            bias_type=BiasType.INTERSECTIONAL,
            description='Compound discrimination on multiple attributes',
            severity=0.9,
            apply_function=self._apply_intersectional_bias,
            parameters={'compound_factor': 1.5}
        )
        
        # Glass ceiling pattern
        patterns['glass_ceiling'] = BiasPattern(
            name='glass_ceiling',
            bias_type=BiasType.GLASS_CEILING,
            description='Bias that increases at higher levels',
            severity=0.7,
            apply_function=self._apply_glass_ceiling,
            parameters={'ceiling_threshold': 0.7, 'ceiling_effect': 0.5}
        )
        
        # Statistical discrimination pattern
        patterns['statistical_discrimination'] = BiasPattern(
            name='statistical_discrimination',
            bias_type=BiasType.STATISTICAL,
            description='Discrimination based on group statistics',
            severity=0.4,
            apply_function=self._apply_statistical_discrimination,
            parameters={'group_penalty': 0.15}
        )
        
        # Historical bias pattern
        patterns['historical_bias'] = BiasPattern(
            name='historical_bias',
            bias_type=BiasType.HISTORICAL,
            description='Past discrimination reflected in current data',
            severity=0.6,
            apply_function=self._apply_historical_bias,
            parameters={'historical_weight': 0.3}
        )
        
        # Sampling bias pattern
        patterns['sampling_bias'] = BiasPattern(
            name='sampling_bias',
            bias_type=BiasType.SAMPLING,
            description='Biased representation in dataset',
            severity=0.5,
            apply_function=self._apply_sampling_bias,
            parameters={'undersampling_rate': 0.3}
        )
        
        # Subtle gradient pattern
        patterns['subtle_gradient'] = BiasPattern(
            name='subtle_gradient',
            bias_type=BiasType.STATISTICAL,
            description='Barely perceptible but consistent bias',
            severity=0.2,
            apply_function=self._apply_subtle_gradient,
            parameters={'gradient_strength': 0.1}
        )
        
        # Temporal bias pattern
        patterns['temporal_bias'] = BiasPattern(
            name='temporal_bias',
            bias_type=BiasType.HISTORICAL,
            description='Bias that varies over time periods',
            severity=0.5,
            apply_function=self._apply_temporal_bias,
            parameters={'time_effect': 0.4}
        )
        
        self.logger.info(f"Initialized {len(patterns)} bias patterns")
        return patterns
    
    def _apply_direct_discrimination(self, df: pd.DataFrame, protected_col: str,
                                    outcome_col: str, params: Dict) -> pd.DataFrame:
        """
        Apply direct discrimination pattern.
        
        Args:
            df: Dataset
            protected_col: Protected attribute column
            outcome_col: Outcome column
            params: Pattern parameters
            
        Returns:
            Modified dataset
        """
        df_copy = df.copy()
        
        # Identify majority and minority groups
        value_counts = df_copy[protected_col].value_counts()
        if len(value_counts) < 2:
            return df_copy
        
        majority = value_counts.index[0]
        minorities = value_counts.index[1:]
        
        # Apply different rates
        for idx, row in df_copy.iterrows():
            if row[protected_col] == majority:
                # Advantage for majority
                if np.random.random() < params['advantage_rate']:
                    df_copy.at[idx, outcome_col] = 1
            elif row[protected_col] in minorities:
                # Disadvantage for minorities
                if np.random.random() < params['disadvantage_rate']:
                    df_copy.at[idx, outcome_col] = 0
        
        return df_copy
    
    def _apply_threshold_shifting(self, df: pd.DataFrame, protected_col: str,
                                 score_col: str, outcome_col: str, params: Dict) -> pd.DataFrame:
        """
        Apply threshold shifting pattern.
        
        Args:
            df: Dataset
            protected_col: Protected attribute column
            score_col: Score/feature column
            outcome_col: Outcome column
            params: Pattern parameters
            
        Returns:
            Modified dataset
        """
        df_copy = df.copy()
        
        # Calculate base threshold
        base_threshold = df_copy[score_col].median()
        
        # Apply different thresholds for different groups
        groups = df_copy[protected_col].unique()
        for i, group in enumerate(groups):
            group_mask = df_copy[protected_col] == group
            
            # Shift threshold for certain groups
            if i % 2 == 0:  # Arbitrary selection of disadvantaged groups
                threshold = base_threshold + params['threshold_difference']
            else:
                threshold = base_threshold - params['threshold_difference']
            
            # Apply threshold
            df_copy.loc[group_mask & (df_copy[score_col] >= threshold), outcome_col] = 1
            df_copy.loc[group_mask & (df_copy[score_col] < threshold), outcome_col] = 0
        
        return df_copy
    
    def _apply_proxy_discrimination(self, df: pd.DataFrame, protected_col: str,
                                   proxy_col: str, outcome_col: str, params: Dict) -> pd.DataFrame:
        """
        Apply proxy discrimination pattern.
        
        Args:
            df: Dataset
            protected_col: Protected attribute column
            proxy_col: Proxy feature column
            outcome_col: Outcome column
            params: Pattern parameters
            
        Returns:
            Modified dataset
        """
        df_copy = df.copy()
        
        # Create correlation between protected attribute and proxy
        protected_values = df_copy[protected_col].unique()
        
        # Map protected values to proxy values with correlation
        for protected_val in protected_values:
            mask = df_copy[protected_col] == protected_val
            
            # Add correlated noise to proxy
            if np.random.random() < params['correlation_strength']:
                df_copy.loc[mask, proxy_col] = df_copy.loc[mask, proxy_col] + np.random.normal(0, 0.1)
        
        # Use proxy for outcome determination
        proxy_threshold = df_copy[proxy_col].median()
        df_copy.loc[df_copy[proxy_col] >= proxy_threshold, outcome_col] = 1
        df_copy.loc[df_copy[proxy_col] < proxy_threshold, outcome_col] = 0
        
        return df_copy
    
    def _apply_intersectional_bias(self, df: pd.DataFrame, protected_cols: List[str],
                                  outcome_col: str, params: Dict) -> pd.DataFrame:
        """
        Apply intersectional bias pattern.
        
        Args:
            df: Dataset
            protected_cols: List of protected attribute columns
            outcome_col: Outcome column
            params: Pattern parameters
            
        Returns:
            Modified dataset
        """
        df_copy = df.copy()
        
        if len(protected_cols) < 2:
            return df_copy
        
        # Create intersection groups
        df_copy['intersection'] = df_copy[protected_cols[0]].astype(str)
        for col in protected_cols[1:]:
            df_copy['intersection'] = df_copy['intersection'] + '_' + df_copy[col].astype(str)
        
        # Identify disadvantaged intersections
        intersection_counts = df_copy['intersection'].value_counts()
        
        # Apply compound discrimination to minority intersections
        for intersection in intersection_counts.index[len(intersection_counts)//2:]:
            mask = df_copy['intersection'] == intersection
            base_rate = df_copy[outcome_col].mean()
            
            # Apply compound factor
            disadvantage_rate = base_rate / params['compound_factor']
            df_copy.loc[mask, outcome_col] = np.random.choice(
                [0, 1], 
                size=mask.sum(),
                p=[1 - disadvantage_rate, disadvantage_rate]
            )
        
        df_copy.drop('intersection', axis=1, inplace=True)
        return df_copy
    
    def _apply_glass_ceiling(self, df: pd.DataFrame, protected_col: str,
                            level_col: str, outcome_col: str, params: Dict) -> pd.DataFrame:
        """
        Apply glass ceiling pattern.
        
        Args:
            df: Dataset
            protected_col: Protected attribute column
            level_col: Level/score column
            outcome_col: Outcome column
            params: Pattern parameters
            
        Returns:
            Modified dataset
        """
        df_copy = df.copy()
        
        # Normalize level column
        level_normalized = (df_copy[level_col] - df_copy[level_col].min()) / \
                          (df_copy[level_col].max() - df_copy[level_col].min())
        
        # Apply ceiling effect for certain groups
        protected_values = df_copy[protected_col].unique()
        disadvantaged_groups = protected_values[::2]  # Every other group
        
        for idx, row in df_copy.iterrows():
            if row[protected_col] in disadvantaged_groups:
                # Check if above ceiling threshold
                if level_normalized.iloc[idx] > params['ceiling_threshold']:
                    # Apply ceiling effect
                    if np.random.random() < params['ceiling_effect']:
                        df_copy.at[idx, outcome_col] = 0
        
        return df_copy
    
    def _apply_statistical_discrimination(self, df: pd.DataFrame, protected_col: str,
                                        outcome_col: str, params: Dict) -> pd.DataFrame:
        """
        Apply statistical discrimination pattern.
        
        Args:
            df: Dataset
            protected_col: Protected attribute column
            outcome_col: Outcome column
            params: Pattern parameters
            
        Returns:
            Modified dataset
        """
        df_copy = df.copy()
        
        # Calculate group statistics
        group_stats = df_copy.groupby(protected_col)[outcome_col].mean()
        
        # Apply penalty based on group statistics
        for group, mean_outcome in group_stats.items():
            if mean_outcome < group_stats.median():
                # Apply penalty to underperforming groups
                mask = df_copy[protected_col] == group
                penalty = params['group_penalty']
                
                for idx in df_copy[mask].index:
                    if np.random.random() < penalty:
                        df_copy.at[idx, outcome_col] = 0
        
        return df_copy
    
    def _apply_historical_bias(self, df: pd.DataFrame, protected_col: str,
                              historical_col: str, outcome_col: str, params: Dict) -> pd.DataFrame:
        """
        Apply historical bias pattern.
        
        Args:
            df: Dataset
            protected_col: Protected attribute column
            historical_col: Historical feature column
            outcome_col: Outcome column
            params: Pattern parameters
            
        Returns:
            Modified dataset
        """
        df_copy = df.copy()
        
        # Historical disadvantage for certain groups
        protected_values = df_copy[protected_col].unique()
        historically_disadvantaged = protected_values[::2]
        
        for idx, row in df_copy.iterrows():
            if row[protected_col] in historically_disadvantaged:
                # Historical factor influences current outcome
                historical_influence = row[historical_col] * params['historical_weight']
                
                # Reduce positive outcome probability
                if np.random.random() < historical_influence:
                    df_copy.at[idx, outcome_col] = 0
        
        return df_copy
    
    def _apply_sampling_bias(self, df: pd.DataFrame, protected_col: str,
                            outcome_col: str, params: Dict) -> pd.DataFrame:
        """
        Apply sampling bias pattern.
        
        Args:
            df: Dataset
            protected_col: Protected attribute column
            outcome_col: Outcome column
            params: Pattern parameters
            
        Returns:
            Modified dataset with sampling bias
        """
        df_copy = df.copy()
        
        # Undersample certain groups
        protected_values = df_copy[protected_col].unique()
        undersampled_groups = protected_values[::2]
        
        indices_to_keep = []
        for group in protected_values:
            group_indices = df_copy[df_copy[protected_col] == group].index
            
            if group in undersampled_groups:
                # Undersample this group
                keep_size = int(len(group_indices) * (1 - params['undersampling_rate']))
                kept_indices = np.random.choice(group_indices, keep_size, replace=False)
            else:
                kept_indices = group_indices
            
            indices_to_keep.extend(kept_indices)
        
        return df_copy.loc[indices_to_keep].reset_index(drop=True)
    
    def _apply_subtle_gradient(self, df: pd.DataFrame, protected_col: str,
                              score_col: str, outcome_col: str, params: Dict) -> pd.DataFrame:
        """
        Apply subtle gradient bias pattern.
        
        Args:
            df: Dataset
            protected_col: Protected attribute column
            score_col: Score column
            outcome_col: Outcome column
            params: Pattern parameters
            
        Returns:
            Modified dataset
        """
        df_copy = df.copy()
        
        # Apply small consistent disadvantage
        protected_values = df_copy[protected_col].unique()
        disadvantaged = protected_values[0]  # Pick one group
        
        mask = df_copy[protected_col] == disadvantaged
        
        # Slightly adjust scores
        df_copy.loc[mask, score_col] = df_copy.loc[mask, score_col] * (1 - params['gradient_strength'])
        
        # Recalculate outcomes based on adjusted scores
        threshold = df_copy[score_col].median()
        df_copy[outcome_col] = (df_copy[score_col] >= threshold).astype(int)
        
        return df_copy
    
    def _apply_temporal_bias(self, df: pd.DataFrame, protected_col: str,
                            time_col: str, outcome_col: str, params: Dict) -> pd.DataFrame:
        """
        Apply temporal bias pattern.
        
        Args:
            df: Dataset
            protected_col: Protected attribute column
            time_col: Time period column
            outcome_col: Outcome column
            params: Pattern parameters
            
        Returns:
            Modified dataset
        """
        df_copy = df.copy()
        
        # Bias varies over time periods
        time_periods = df_copy[time_col].unique()
        protected_values = df_copy[protected_col].unique()
        
        for period in time_periods:
            for i, protected_val in enumerate(protected_values):
                mask = (df_copy[time_col] == period) & (df_copy[protected_col] == protected_val)
                
                # Oscillating bias pattern
                time_factor = np.sin(period * np.pi / len(time_periods))
                
                if i % 2 == 0:  # Disadvantaged groups
                    bias_strength = params['time_effect'] * abs(time_factor)
                    
                    for idx in df_copy[mask].index:
                        if np.random.random() < bias_strength:
                            df_copy.at[idx, outcome_col] = 0
        
        return df_copy
    
    def get_random_patterns(self, n: int = 3) -> List[BiasPattern]:
        """
        Get random selection of bias patterns.
        
        Args:
            n: Number of patterns to select
            
        Returns:
            List of selected bias patterns
        """
        pattern_names = list(self.patterns.keys())
        selected_names = np.random.choice(pattern_names, min(n, len(pattern_names)), replace=False)
        return [self.patterns[name] for name in selected_names]
    
    def get_pattern_by_severity(self, min_severity: float = 0.0, 
                               max_severity: float = 1.0) -> List[BiasPattern]:
        """
        Get patterns within severity range.
        
        Args:
            min_severity: Minimum severity
            max_severity: Maximum severity
            
        Returns:
            List of patterns in severity range
        """
        return [p for p in self.patterns.values() 
                if min_severity <= p.severity <= max_severity]
    
    def get_pattern_combinations(self) -> List[List[str]]:
        """
        Get recommended pattern combinations for realistic bias.
        
        Returns:
            List of pattern name combinations
        """
        combinations = [
            ['direct_discrimination'],
            ['threshold_shifting', 'statistical_discrimination'],
            ['proxy_discrimination', 'historical_bias'],
            ['intersectional_bias', 'glass_ceiling'],
            ['subtle_gradient', 'temporal_bias'],
            ['sampling_bias', 'statistical_discrimination'],
            ['direct_discrimination', 'proxy_discrimination', 'historical_bias'],
            ['threshold_shifting', 'glass_ceiling', 'statistical_discrimination'],
            []  # No bias (control group)
        ]
        
        return combinations

# Global pattern library instance
pattern_library = PatternLibrary()

if __name__ == "__main__":
    print("Pattern Library Test")
    print("-" * 50)
    
    # List all patterns
    print(f"Total patterns: {len(pattern_library.patterns)}")
    print("\nAvailable patterns:")
    for name, pattern in pattern_library.patterns.items():
        print(f"  - {name}: {pattern.description} (severity: {pattern.severity})")
    
    # Get random patterns
    random_patterns = pattern_library.get_random_patterns(3)
    print(f"\nRandom selection of {len(random_patterns)} patterns:")
    for pattern in random_patterns:
        print(f"  - {pattern.name}: {pattern.bias_type.value}")
    
    # Get patterns by severity
    high_severity = pattern_library.get_pattern_by_severity(0.7, 1.0)
    print(f"\nHigh severity patterns (>0.7): {len(high_severity)}")
    for pattern in high_severity:
        print(f"  - {pattern.name}: {pattern.severity}")
    
    # Get pattern combinations
    combinations = pattern_library.get_pattern_combinations()
    print(f"\nRecommended pattern combinations: {len(combinations)}")
    for i, combo in enumerate(combinations[:5]):
        print(f"  {i+1}. {' + '.join(combo) if combo else 'No bias (control)'}")
    
    print("\nPattern library test completed")