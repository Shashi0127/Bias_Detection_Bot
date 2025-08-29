"""
Statistical analysis engine for bias detection.
Provides statistical tests and fairness metrics that work without ML models.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from config.settings import (
    DISPARATE_IMPACT_THRESHOLD, STATISTICAL_SIGNIFICANCE,
    BIAS_THRESHOLD
)
from src.utils.logger import get_logger

class StatisticalEngine:
    """
    Performs statistical analysis to detect bias patterns in data.
    """
    
    def __init__(self):
        self.logger = get_logger('statistical_engine')
    
    def analyze_bias(self, df: pd.DataFrame, protected_attributes: List[str],
                     outcome_variables: List[str]) -> Dict[str, Any]:
        """
        Comprehensive statistical bias analysis.
        
        Args:
            df: Dataset
            protected_attributes: List of protected attribute columns
            outcome_variables: List of outcome variable columns
            
        Returns:
            Dictionary with statistical analysis results
        """
        self.logger.info(f"Starting statistical analysis for {len(protected_attributes)} "
                        f"protected attributes and {len(outcome_variables)} outcomes")
        
        results = {
            'disparate_impact': {},
            'statistical_tests': {},
            'fairness_metrics': {},
            'group_statistics': {},
            'overall_bias_score': 0,
            'bias_detected': False,
            'confidence': 0
        }
        
        if not protected_attributes or not outcome_variables:
            self.logger.warning("No protected attributes or outcomes found")
            return results
        
        bias_scores = []
        
        for outcome in outcome_variables:
            for protected in protected_attributes:
                # Skip if columns don't exist
                if protected not in df.columns or outcome not in df.columns:
                    continue
                
                # Disparate impact analysis
                impact = self._calculate_disparate_impact(df, protected, outcome)
                if impact:
                    results['disparate_impact'][f"{protected}_vs_{outcome}"] = impact
                    if impact['bias_detected']:
                        bias_scores.append(impact['severity'])
                
                # Statistical independence tests
                stat_test = self._perform_statistical_test(df, protected, outcome)
                if stat_test:
                    results['statistical_tests'][f"{protected}_vs_{outcome}"] = stat_test
                    if stat_test['significant']:
                        bias_scores.append(0.5)  # Weight statistical significance
                
                # Fairness metrics
                fairness = self._calculate_fairness_metrics(df, protected, outcome)
                if fairness:
                    results['fairness_metrics'][f"{protected}_vs_{outcome}"] = fairness
                
                # Group statistics
                group_stats = self._calculate_group_statistics(df, protected, outcome)
                if group_stats:
                    results['group_statistics'][f"{protected}_vs_{outcome}"] = group_stats
        
        # Calculate overall bias score
        if bias_scores:
            results['overall_bias_score'] = np.mean(bias_scores)
            results['bias_detected'] = results['overall_bias_score'] > BIAS_THRESHOLD
            results['confidence'] = self._calculate_confidence(results)
        
        self.logger.info(f"Statistical analysis complete. Bias detected: {results['bias_detected']}, "
                        f"Score: {results['overall_bias_score']:.2f}")
        
        return results
    
    def _calculate_disparate_impact(self, df: pd.DataFrame, protected: str, 
                                   outcome: str) -> Optional[Dict[str, Any]]:
        """
        Calculate disparate impact ratio (80% rule).
        
        Args:
            df: Dataset
            protected: Protected attribute column
            outcome: Outcome variable column
            
        Returns:
            Disparate impact analysis results
        """
        try:
            # Group by protected attribute and calculate outcome rates
            grouped = df.groupby(protected)[outcome].agg(['mean', 'count'])
            grouped = grouped[grouped['count'] >= 10]  # Minimum sample size
            
            if len(grouped) < 2:
                return None
            
            # Find majority and minority groups
            majority_group = grouped['count'].idxmax()
            majority_rate = grouped.loc[majority_group, 'mean']
            
            impact_ratios = {}
            min_ratio = 1.0
            
            for group in grouped.index:
                if group != majority_group:
                    group_rate = grouped.loc[group, 'mean']
                    if majority_rate > 0:
                        ratio = group_rate / majority_rate
                        impact_ratios[group] = ratio
                        min_ratio = min(min_ratio, ratio)
            
            # Determine bias based on 80% rule
            bias_detected = min_ratio < DISPARATE_IMPACT_THRESHOLD
            severity = 1.0 - min_ratio if bias_detected else 0
            
            return {
                'majority_group': majority_group,
                'majority_rate': float(majority_rate),
                'impact_ratios': impact_ratios,
                'min_ratio': float(min_ratio),
                'bias_detected': bias_detected,
                'severity': float(severity),
                'threshold': DISPARATE_IMPACT_THRESHOLD
            }
            
        except Exception as e:
            self.logger.error(f"Disparate impact calculation failed: {str(e)}")
            return None
    
    def _perform_statistical_test(self, df: pd.DataFrame, protected: str, 
                                 outcome: str) -> Optional[Dict[str, Any]]:
        """
        Perform appropriate statistical test for independence.
        
        Args:
            df: Dataset
            protected: Protected attribute column
            outcome: Outcome variable column
            
        Returns:
            Statistical test results
        """
        try:
            # Determine test type based on data types
            protected_numeric = pd.api.types.is_numeric_dtype(df[protected])
            outcome_numeric = pd.api.types.is_numeric_dtype(df[outcome])
            
            if not protected_numeric and not outcome_numeric:
                # Both categorical - Chi-square test
                return self._chi_square_test(df, protected, outcome)
            elif protected_numeric and outcome_numeric:
                # Both numerical - Correlation test
                return self._correlation_test(df, protected, outcome)
            elif not protected_numeric and outcome_numeric:
                # Categorical vs Numerical - ANOVA or t-test
                return self._anova_test(df, protected, outcome)
            else:
                # Numerical vs Categorical - Logistic regression coefficient
                return self._logistic_coefficient_test(df, protected, outcome)
                
        except Exception as e:
            self.logger.error(f"Statistical test failed: {str(e)}")
            return None
    
    def _chi_square_test(self, df: pd.DataFrame, var1: str, var2: str) -> Dict[str, Any]:
        """
        Perform chi-square test of independence.
        
        Args:
            df: Dataset
            var1, var2: Variable columns
            
        Returns:
            Chi-square test results
        """
        contingency = pd.crosstab(df[var1], df[var2])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        return {
            'test_type': 'chi_square',
            'statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'significant': p_value < STATISTICAL_SIGNIFICANCE,
            'interpretation': 'Variables are dependent' if p_value < STATISTICAL_SIGNIFICANCE 
                            else 'Variables are independent'
        }
    
    def _correlation_test(self, df: pd.DataFrame, var1: str, var2: str) -> Dict[str, Any]:
        """
        Perform correlation test for numerical variables.
        
        Args:
            df: Dataset
            var1, var2: Variable columns
            
        Returns:
            Correlation test results
        """
        # Remove missing values
        clean_df = df[[var1, var2]].dropna()
        correlation, p_value = stats.pearsonr(clean_df[var1], clean_df[var2])
        
        return {
            'test_type': 'correlation',
            'statistic': float(correlation),
            'p_value': float(p_value),
            'significant': p_value < STATISTICAL_SIGNIFICANCE,
            'interpretation': f"Correlation: {correlation:.3f}"
        }
    
    def _anova_test(self, df: pd.DataFrame, categorical: str, 
                   numerical: str) -> Dict[str, Any]:
        """
        Perform ANOVA test.
        
        Args:
            df: Dataset
            categorical: Categorical variable column
            numerical: Numerical variable column
            
        Returns:
            ANOVA test results
        """
        groups = []
        for category in df[categorical].unique():
            group_data = df[df[categorical] == category][numerical].dropna()
            if len(group_data) > 0:
                groups.append(group_data)
        
        if len(groups) < 2:
            return None
        
        f_stat, p_value = stats.f_oneway(*groups)
        
        return {
            'test_type': 'anova',
            'statistic': float(f_stat),
            'p_value': float(p_value),
            'significant': p_value < STATISTICAL_SIGNIFICANCE,
            'interpretation': 'Groups have different means' if p_value < STATISTICAL_SIGNIFICANCE 
                            else 'Groups have similar means'
        }
    
    def _logistic_coefficient_test(self, df: pd.DataFrame, numerical: str, 
                                  categorical: str) -> Dict[str, Any]:
        """
        Test significance of numerical variable in predicting categorical outcome.
        
        Args:
            df: Dataset
            numerical: Numerical variable column
            categorical: Categorical variable column
            
        Returns:
            Coefficient test results
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data
        X = df[[numerical]].dropna()
        y = df.loc[X.index, categorical]
        
        # Handle binary encoding
        if y.nunique() == 2:
            y = pd.get_dummies(y, drop_first=True).iloc[:, 0]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit logistic regression
        lr = LogisticRegression(random_state=42)
        lr.fit(X_scaled, y)
        
        # Get coefficient and calculate z-score
        coef = lr.coef_[0][0]
        # Simple approximation of standard error
        se = 1.0 / np.sqrt(len(X))
        z_score = coef / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            'test_type': 'logistic_coefficient',
            'statistic': float(coef),
            'z_score': float(z_score),
            'p_value': float(p_value),
            'significant': p_value < STATISTICAL_SIGNIFICANCE,
            'interpretation': f"Coefficient: {coef:.3f}"
        }
    
    def _calculate_fairness_metrics(self, df: pd.DataFrame, protected: str, 
                                   outcome: str) -> Dict[str, Any]:
        """
        Calculate various fairness metrics.
        
        Args:
            df: Dataset
            protected: Protected attribute column
            outcome: Outcome variable column
            
        Returns:
            Fairness metrics
        """
        try:
            metrics = {}
            
            # Group statistics
            grouped = df.groupby(protected)[outcome].agg(['mean', 'count'])
            
            # Demographic parity difference
            max_rate = grouped['mean'].max()
            min_rate = grouped['mean'].min()
            metrics['demographic_parity_difference'] = float(max_rate - min_rate)
            
            # Statistical parity ratio
            if min_rate > 0:
                metrics['statistical_parity_ratio'] = float(min_rate / max_rate)
            else:
                metrics['statistical_parity_ratio'] = 0.0
            
            # Equal opportunity difference (for positive outcomes)
            positive_df = df[df[outcome] == 1] if outcome in df.columns else df
            if len(positive_df) > 0:
                pos_grouped = positive_df.groupby(protected).size() / df.groupby(protected).size()
                metrics['equal_opportunity_difference'] = float(pos_grouped.max() - pos_grouped.min())
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Fairness metrics calculation failed: {str(e)}")
            return {}
    
    def _calculate_group_statistics(self, df: pd.DataFrame, protected: str, 
                                   outcome: str) -> Dict[str, Any]:
        """
        Calculate detailed statistics for each group.
        
        Args:
            df: Dataset
            protected: Protected attribute column
            outcome: Outcome variable column
            
        Returns:
            Group statistics
        """
        try:
            group_stats = {}
            
            for group in df[protected].unique():
                group_df = df[df[protected] == group]
                
                if pd.api.types.is_numeric_dtype(df[outcome]):
                    stats_dict = {
                        'count': len(group_df),
                        'outcome_mean': float(group_df[outcome].mean()),
                        'outcome_std': float(group_df[outcome].std()),
                        'outcome_median': float(group_df[outcome].median())
                    }
                else:
                    # For categorical outcomes, calculate distribution
                    value_counts = group_df[outcome].value_counts(normalize=True)
                    stats_dict = {
                        'count': len(group_df),
                        'outcome_distribution': value_counts.to_dict()
                    }
                
                group_stats[str(group)] = stats_dict
            
            return group_stats
            
        except Exception as e:
            self.logger.error(f"Group statistics calculation failed: {str(e)}")
            return {}
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """
        Calculate confidence score for bias detection.
        
        Args:
            results: Analysis results
            
        Returns:
            Confidence score (0-1)
        """
        confidence_factors = []
        
        # Factor 1: Number of significant statistical tests
        significant_tests = sum(1 for test in results['statistical_tests'].values() 
                               if test.get('significant', False))
        if results['statistical_tests']:
            confidence_factors.append(significant_tests / len(results['statistical_tests']))
        
        # Factor 2: Severity of disparate impact
        if results['disparate_impact']:
            severities = [impact.get('severity', 0) for impact in results['disparate_impact'].values()]
            if severities:
                confidence_factors.append(np.mean(severities))
        
        # Factor 3: Consistency across multiple protected attributes
        if len(results['disparate_impact']) > 1:
            bias_detected_count = sum(1 for impact in results['disparate_impact'].values() 
                                     if impact.get('bias_detected', False))
            confidence_factors.append(bias_detected_count / len(results['disparate_impact']))
        
        return np.mean(confidence_factors) if confidence_factors else 0.0

# Convenience function
def analyze_statistical_bias(df: pd.DataFrame, protected_attributes: List[str], 
                           outcome_variables: List[str]) -> Dict[str, Any]:
    """
    Perform statistical bias analysis.
    
    Args:
        df: Dataset
        protected_attributes: Protected attribute columns
        outcome_variables: Outcome columns
        
    Returns:
        Statistical analysis results
    """
    engine = StatisticalEngine()