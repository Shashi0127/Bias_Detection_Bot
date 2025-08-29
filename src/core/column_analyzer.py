"""
Column analyzer for automatic detection of column types in any CSV.
Identifies protected attributes, outcome variables, and features.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter

from config.settings import PROTECTED_ATTRIBUTES, OUTCOME_PATTERNS
from src.utils.logger import get_logger

class ColumnAnalyzer:
    """
    Analyzes CSV columns to automatically detect their types and purposes.
    """
    
    def __init__(self):
        self.logger = get_logger('column_analyzer')
        self.protected_patterns = self._compile_patterns(PROTECTED_ATTRIBUTES)
        self.outcome_patterns = self._compile_patterns(OUTCOME_PATTERNS)
    
    def _compile_patterns(self, pattern_list: List[str]) -> List[re.Pattern]:
        """
        Compile regex patterns for efficient matching.
        
        Args:
            pattern_list: List of pattern strings
            
        Returns:
            List of compiled regex patterns
        """
        return [re.compile(f'.*{pattern}.*', re.IGNORECASE) for pattern in pattern_list]
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive analysis of dataset columns.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with column classifications and metadata
        """
        self.logger.info(f"Analyzing dataset with shape {df.shape}")
        
        analysis = {
            'protected_attributes': [],
            'outcome_variables': [],
            'features': [],
            'metadata': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
            }
        }
        
        for column in df.columns:
            col_info = self._analyze_column(df[column], column)
            
            # Classify column
            if col_info['is_protected']:
                analysis['protected_attributes'].append(col_info)
            elif col_info['is_outcome']:
                analysis['outcome_variables'].append(col_info)
            else:
                analysis['features'].append(col_info)
        
        # Add analysis summary
        analysis['summary'] = self._generate_summary(analysis)
        
        self.logger.info(f"Found {len(analysis['protected_attributes'])} protected attributes, "
                        f"{len(analysis['outcome_variables'])} outcome variables, "
                        f"{len(analysis['features'])} features")
        
        return analysis
    
    def _analyze_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        Analyze individual column characteristics.
        
        Args:
            series: Pandas series (column data)
            column_name: Name of the column
            
        Returns:
            Dictionary with column information
        """
        col_info = {
            'name': column_name,
            'dtype': str(series.dtype),
            'unique_values': series.nunique(),
            'missing_count': series.isnull().sum(),
            'missing_percentage': (series.isnull().sum() / len(series) * 100),
            'is_protected': False,
            'is_outcome': False,
            'protected_type': None,
            'data_type': None,
            'statistics': {}
        }
        
        # Determine data type
        if pd.api.types.is_numeric_dtype(series):
            col_info['data_type'] = 'numerical'
            col_info['statistics'] = {
                'mean': series.mean() if not series.isnull().all() else None,
                'median': series.median() if not series.isnull().all() else None,
                'std': series.std() if not series.isnull().all() else None,
                'min': series.min() if not series.isnull().all() else None,
                'max': series.max() if not series.isnull().all() else None
            }
        else:
            col_info['data_type'] = 'categorical'
            value_counts = series.value_counts()
            col_info['statistics'] = {
                'top_values': value_counts.head(5).to_dict(),
                'category_count': len(value_counts)
            }
        
        # Check if protected attribute
        col_info['is_protected'], col_info['protected_type'] = self._is_protected_attribute(
            column_name, series
        )
        
        # Check if outcome variable
        col_info['is_outcome'] = self._is_outcome_variable(column_name, series)
        
        return col_info
    
    def _is_protected_attribute(self, column_name: str, series: pd.Series) -> Tuple[bool, Optional[str]]:
        """
        Determine if column is a protected attribute.
        
        Args:
            column_name: Name of the column
            series: Column data
            
        Returns:
            Tuple of (is_protected, protected_type)
        """
        column_lower = column_name.lower()
        
        # Direct pattern matching
        for pattern in self.protected_patterns:
            if pattern.match(column_lower):
                # Determine specific type
                if any(term in column_lower for term in ['race', 'ethnic']):
                    return True, 'race/ethnicity'
                elif any(term in column_lower for term in ['gender', 'sex']):
                    return True, 'gender'
                elif any(term in column_lower for term in ['age', 'birth', 'dob']):
                    return True, 'age'
                elif any(term in column_lower for term in ['relig']):
                    return True, 'religion'
                elif any(term in column_lower for term in ['disab', 'handicap']):
                    return True, 'disability'
                elif any(term in column_lower for term in ['national', 'citizen']):
                    return True, 'nationality'
                else:
                    return True, 'other'
        
        # Check for common demographic values in categorical columns
        if series.dtype == 'object' and series.nunique() < 20:
            values_lower = [str(v).lower() for v in series.dropna().unique()]
            
            # Gender detection by values
            gender_values = ['male', 'female', 'm', 'f', 'man', 'woman']
            if any(val in values_lower for val in gender_values):
                return True, 'gender'
            
            # Race/ethnicity detection by values
            race_values = ['white', 'black', 'asian', 'hispanic', 'caucasian', 'african']
            if any(val in values_lower for val in race_values):
                return True, 'race/ethnicity'
        
        return False, None
    
    def _is_outcome_variable(self, column_name: str, series: pd.Series) -> bool:
        """
        Determine if column is an outcome variable.
        
        Args:
            column_name: Name of the column
            series: Column data
            
        Returns:
            True if column appears to be an outcome variable
        """
        column_lower = column_name.lower()
        
        # Check pattern matching
        for pattern in self.outcome_patterns:
            if pattern.match(column_lower):
                # Verify it looks like an outcome (binary or low cardinality)
                if series.nunique() <= 10 or self._is_binary_column(series):
                    return True
        
        # Check for binary columns with suggestive names
        if self._is_binary_column(series):
            suggestive_terms = ['is_', 'has_', 'was_', 'flag', 'indicator']
            if any(term in column_lower for term in suggestive_terms):
                return True
        
        return False
    
    def _is_binary_column(self, series: pd.Series) -> bool:
        """
        Check if column is binary (two unique values).
        
        Args:
            series: Column data
            
        Returns:
            True if column is binary
        """
        unique_values = series.dropna().unique()
        if len(unique_values) == 2:
            # Check for common binary patterns
            if series.dtype in ['int64', 'float64']:
                if set(unique_values).issubset({0, 1, 0.0, 1.0}):
                    return True
            elif series.dtype == 'object':
                values_lower = [str(v).lower() for v in unique_values]
                binary_pairs = [
                    {'yes', 'no'}, {'y', 'n'},
                    {'true', 'false'}, {'t', 'f'},
                    {'1', '0'}, {'pass', 'fail'},
                    {'approved', 'rejected'}, {'hired', 'not hired'}
                ]
                for pair in binary_pairs:
                    if set(values_lower) == pair:
                        return True
        return False
    
    def _generate_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate analysis summary.
        
        Args:
            analysis: Column analysis results
            
        Returns:
            Summary dictionary
        """
        summary = {
            'has_protected_attributes': len(analysis['protected_attributes']) > 0,
            'has_outcome_variables': len(analysis['outcome_variables']) > 0,
            'bias_detection_ready': False,
            'warnings': [],
            'recommendations': []
        }
        
        # Check if ready for bias detection
        if summary['has_protected_attributes'] and summary['has_outcome_variables']:
            summary['bias_detection_ready'] = True
        else:
            if not summary['has_protected_attributes']:
                summary['warnings'].append("No protected attributes detected")
                summary['recommendations'].append(
                    "Consider identifying demographic columns manually"
                )
            if not summary['has_outcome_variables']:
                summary['warnings'].append("No clear outcome variables detected")
                summary['recommendations'].append(
                    "Consider identifying decision/outcome columns manually"
                )
        
        # Check data quality
        missing_pct = analysis['metadata']['missing_percentage']
        if missing_pct > 20:
            summary['warnings'].append(f"High missing data rate: {missing_pct:.1f}%")
            summary['recommendations'].append("Consider data imputation or cleaning")
        
        return summary
    
    def detect_proxy_variables(self, df: pd.DataFrame, 
                              protected_cols: List[str]) -> List[Dict[str, Any]]:
        """
        Detect potential proxy variables for protected attributes.
        
        Args:
            df: Dataframe
            protected_cols: List of protected attribute columns
            
        Returns:
            List of potential proxy variables
        """
        proxies = []
        
        for col in df.columns:
            if col in protected_cols:
                continue
            
            # Check correlation with protected attributes
            for protected in protected_cols:
                if df[protected].dtype in ['object', 'category'] and df[col].dtype in ['object', 'category']:
                    # Categorical correlation using Cramér's V
                    correlation = self._cramers_v(df[protected], df[col])
                    if correlation > 0.5:
                        proxies.append({
                            'proxy_column': col,
                            'protected_attribute': protected,
                            'correlation': correlation,
                            'type': 'categorical'
                        })
                elif pd.api.types.is_numeric_dtype(df[protected]) and pd.api.types.is_numeric_dtype(df[col]):
                    # Numerical correlation
                    correlation = abs(df[protected].corr(df[col]))
                    if correlation > 0.5:
                        proxies.append({
                            'proxy_column': col,
                            'protected_attribute': protected,
                            'correlation': correlation,
                            'type': 'numerical'
                        })
        
        return proxies
    
    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """
        Calculate Cramér's V for categorical correlation.
        
        Args:
            x, y: Categorical series
            
        Returns:
            Cramér's V statistic (0 to 1)
        """
        from scipy.stats import chi2_contingency
        
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        r, k = confusion_matrix.shape
        
        return np.sqrt(chi2 / (n * (min(k, r) - 1))) if n > 0 else 0

# Convenience functions
def analyze_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze dataset columns to detect types.
    
    Args:
        df: Input dataframe
        
    Returns:
        Column analysis results
    """
    analyzer = ColumnAnalyzer()
    return analyzer.analyze_dataset(df)

def get_protected_attributes(df: pd.DataFrame) -> List[str]:
    """
    Get list of protected attribute column names.
    
    Args:
        df: Input dataframe
        
    Returns:
        List of protected attribute column names
    """
    analyzer = ColumnAnalyzer()
    analysis = analyzer.analyze_dataset(df)
    return [col['name'] for col in analysis['protected_attributes']]

def get_outcome_variables(df: pd.DataFrame) -> List[str]:
    """
    Get list of outcome variable column names.
    
    Args:
        df: Input dataframe
        
    Returns:
        List of outcome variable column names
    """
    analyzer = ColumnAnalyzer()
    analysis = analyzer.analyze_dataset(df)
    return [col['name'] for col in analysis['outcome_variables']]

if __name__ == "__main__":
    print("Column Analyzer Test")
    print("-" * 50)
    
    # Create test dataset
    test_df = pd.DataFrame({
        'age': np.random.randint(20, 60, 100),
        'gender': np.random.choice(['Male', 'Female'], 100),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], 100),
        'income': np.random.normal(50000, 15000, 100),
        'credit_score': np.random.randint(300, 850, 100),
        'loan_approved': np.random.choice([0, 1], 100),
        'risk_score': np.random.uniform(0, 1, 100)
    })
    
    # Analyze columns
    analyzer = ColumnAnalyzer()
    results = analyzer.analyze_dataset(test_df)
    
    print(f"Protected Attributes: {[c['name'] for c in results['protected_attributes']]}")
    print(f"Outcome Variables: {[c['name'] for c in results['outcome_variables']]}")
    print(f"Features: {[c['name'] for c in results['features']]}")
    print(f"\nSummary:")
    print(f"  Ready for bias detection: {results['summary']['bias_detection_ready']}")
    
    if results['summary']['warnings']:
        print(f"  Warnings: {results['summary']['warnings']}")
    
    # Test proxy detection
    protected = get_protected_attributes(test_df)
    proxies = analyzer.detect_proxy_variables(test_df, protected)
    if proxies:
        print(f"\nPotential proxy variables detected:")
        for proxy in proxies:
            print(f"  {proxy['proxy_column']} -> {proxy['protected_attribute']} "
                  f"(correlation: {proxy['correlation']:.2f})")
    
    print("\nColumn analyzer test completed")