"""Auto-generate insights and suggestions from data analysis."""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from .profiler import StatisticalProfiler
from .type_detector import DataTypeDetector


class InsightEngine:
    """Generate automated insights and recommendations."""
    
    SEVERITY_CRITICAL = "critical"
    SEVERITY_HIGH = "high"
    SEVERITY_MEDIUM = "medium"
    SEVERITY_LOW = "low"
    SEVERITY_INFO = "info"
    
    def __init__(self, profiler: StatisticalProfiler):
        self.profiler = profiler
        self.profile = profiler.get_profile()
        self.insights: List[Dict[str, Any]] = []
        self.suggestions: List[Dict[str, Any]] = []
        self._generate_all()
    
    def _generate_all(self):
        """Generate all insights and suggestions."""
        self._check_missing_values()
        self._check_duplicates()
        self._check_data_quality()
        self._check_target_variable()
        self._check_correlations()
        self._check_class_imbalance()
        self._check_outliers()
        self._check_data_types()
        self._check_high_cardinality()
        self._check_constant_columns()
    
    def _add_insight(self, category: str, message: str, severity: str, 
                     details: Optional[Dict] = None):
        """Add an insight."""
        self.insights.append({
            'category': category,
            'message': message,
            'severity': severity,
            'details': details or {}
        })
    
    def _add_suggestion(self, action: str, reason: str, priority: str,
                       implementation: Optional[str] = None):
        """Add a suggestion."""
        self.suggestions.append({
            'action': action,
            'reason': reason,
            'priority': priority,
            'implementation': implementation
        })
    
    def _check_missing_values(self):
        """Check for missing value issues."""
        missing_analysis = self.profile.get('missing_values', {})
        
        if missing_analysis.get('total_missing', 0) == 0:
            self._add_insight(
                'missing_values',
                'Dataset has no missing values - excellent data completeness!',
                self.SEVERITY_INFO
            )
            return
        
        # Critical missing
        critical_cols = [
            col for col, info in missing_analysis.get('missing_by_column', {}).items()
            if info['severity'] == 'critical'
        ]
        if critical_cols:
            self._add_insight(
                'missing_values',
                f'Found {len(critical_cols)} columns with >50% missing values',
                self.SEVERITY_CRITICAL,
                {'columns': critical_cols}
            )
            for col in critical_cols[:3]:  # Limit suggestions
                self._add_suggestion(
                    f'Consider dropping column "{col}"',
                    f'Column has >50% missing values, may not be useful',
                    'high',
                    f"df = df.drop('{col}', axis=1)"
                )
        
        # High missing
        high_cols = [
            col for col, info in missing_analysis.get('missing_by_column', {}).items()
            if info['severity'] == 'high'
        ]
        if high_cols:
            self._add_insight(
                'missing_values',
                f'{len(high_cols)} columns have 20-50% missing values',
                self.SEVERITY_HIGH,
                {'columns': high_cols}
            )
            self._add_suggestion(
                'Apply imputation for columns with moderate missing values',
                '20-50% missing - imputation recommended before modeling',
                'high',
                'Use SimpleImputer, KNNImputer, or domain-specific imputation'
            )
        
        # Moderate missing
        moderate_cols = [
            col for col, info in missing_analysis.get('missing_by_column', {}).items()
            if info['severity'] == 'moderate'
        ]
        if moderate_cols:
            self._add_insight(
                'missing_values',
                f'{len(moderate_cols)} columns have 5-20% missing values',
                self.SEVERITY_MEDIUM,
                {'columns': moderate_cols}
            )
            self._add_suggestion(
                'Consider imputation for columns with low-moderate missing values',
                '5-20% missing - simple imputation (mean/median/mode) may suffice',
                'medium'
            )
    
    def _check_duplicates(self):
        """Check for duplicate rows."""
        dup_analysis = self.profile.get('duplicates', {})
        dup_count = dup_analysis.get('total_duplicates', 0)
        dup_pct = dup_analysis.get('duplicate_percentage', 0)
        
        if dup_count == 0:
            self._add_insight(
                'duplicates',
                'No duplicate rows found',
                self.SEVERITY_INFO
            )
        elif dup_pct > 10:
            self._add_insight(
                'duplicates',
                f'Found {dup_count} duplicate rows ({dup_pct:.1f}%) - critical issue',
                self.SEVERITY_CRITICAL,
                {'count': dup_count, 'percentage': dup_pct}
            )
            self._add_suggestion(
                'Remove duplicate rows immediately',
                f'{dup_pct:.1f}% duplicates will bias models',
                'critical',
                'df = df.drop_duplicates()'
            )
        elif dup_pct > 1:
            self._add_insight(
                'duplicates',
                f'Found {dup_count} duplicate rows ({dup_pct:.1f}%)',
                self.SEVERITY_MEDIUM,
                {'count': dup_count, 'percentage': dup_pct}
            )
            self._add_suggestion(
                'Review and remove duplicate rows',
                'Duplicates may exist due to data collection issues',
                'medium',
                'df = df.drop_duplicates()'
            )
    
    def _check_data_quality(self):
        """Check overall data quality score."""
        quality = self.profile.get('quality_score', {})
        overall = quality.get('overall', 0)
        
        if overall >= 90:
            self._add_insight(
                'quality',
                f'Excellent data quality score: {overall}%',
                self.SEVERITY_INFO,
                quality
            )
        elif overall >= 70:
            self._add_insight(
                'quality',
                f'Good data quality score: {overall}%',
                self.SEVERITY_LOW,
                quality
            )
        elif overall >= 50:
            self._add_insight(
                'quality',
                f'Moderate data quality score: {overall}% - needs attention',
                self.SEVERITY_MEDIUM,
                quality
            )
        else:
            self._add_insight(
                'quality',
                f'Poor data quality score: {overall}% - significant issues detected',
                self.SEVERITY_CRITICAL,
                quality
            )
    
    def _check_target_variable(self):
        """Analyze potential target variable."""
        target = self.profile.get('target_suggestion')
        if not target:
            self._add_insight(
                'target',
                'No clear target variable detected',
                self.SEVERITY_INFO
            )
            return
        
        target_info = self.profile.get('columns', {}).get(target, {})
        target_type = target_info.get('type_info', {}).get('type', 'unknown')
        stats = target_info.get('statistics', {})
        
        if target_type in [DataTypeDetector.CATEGORICAL_NOMINAL, DataTypeDetector.BOOLEAN, DataTypeDetector.CATEGORICAL_ORDINAL]:
            unique_count = stats.get('unique', 0)
            if unique_count == 2:
                self._add_insight(
                    'target',
                    f'Column "{target}" appears to be a BINARY CLASSIFICATION target',
                    self.SEVERITY_INFO,
                    {'type': 'binary_classification', 'target': target}
                )
            elif unique_count <= 20:
                self._add_insight(
                    'target',
                    f'Column "{target}" appears to be a MULTI-CLASS CLASSIFICATION target ({unique_count} classes)',
                    self.SEVERITY_INFO,
                    {'type': 'multiclass_classification', 'target': target, 'n_classes': unique_count}
                )
            else:
                self._add_insight(
                    'target',
                    f'Column "{target}" has high cardinality ({unique_count}) - consider as regression or reduce categories',
                    self.SEVERITY_MEDIUM,
                    {'type': 'high_cardinality_target', 'target': target}
                )
        elif target_type == DataTypeDetector.NUMERIC_CONTINUOUS:
            self._add_insight(
                'target',
                f'Column "{target}" appears to be a REGRESSION target',
                self.SEVERITY_INFO,
                {'type': 'regression', 'target': target}
            )
        
        # Check for class imbalance in classification
        if target_type in [DataTypeDetector.CATEGORICAL_NOMINAL, DataTypeDetector.BOOLEAN]:
            balance = stats.get('balance_score', 0)
            if balance < 0.5:
                self._add_insight(
                    'target',
                    f'Target "{target}" shows significant class imbalance',
                    self.SEVERITY_HIGH,
                    {'balance_score': balance}
                )
                self._add_suggestion(
                    f'Address class imbalance in target "{target}"',
                    'Imbalanced classes can lead to biased models',
                    'high',
                    'Use SMOTE, class weights, or stratified sampling'
                )
    
    def _check_correlations(self):
        """Analyze feature correlations."""
        corr_analysis = self.profile.get('correlations', {})
        if not corr_analysis.get('available'):
            return
        
        high_corr = corr_analysis.get('high_correlations', [])
        
        if not high_corr:
            self._add_insight(
                'correlations',
                'No high correlations (>0.7) detected between features',
                self.SEVERITY_INFO
            )
            return
        
        n_strong = sum(1 for c in high_corr if c['strength'] == 'strong')
        
        if n_strong > 0:
            self._add_insight(
                'correlations',
                f'Found {n_strong} strongly correlated feature pairs (>0.8) - potential multicollinearity',
                self.SEVERITY_HIGH,
                {'strong_correlations': [c for c in high_corr if c['strength'] == 'strong'][:5]}
            )
            self._add_suggestion(
                'Remove or combine highly correlated features',
                'Multicollinearity can cause model instability',
                'high',
                'Use VIF analysis or drop one of each correlated pair'
            )
        
        if len(high_corr) > n_strong:
            self._add_insight(
                'correlations',
                f'Found {len(high_corr) - n_strong} moderately correlated feature pairs (0.7-0.8)',
                self.SEVERITY_MEDIUM
            )
    
    def _check_class_imbalance(self):
        """Check for class imbalance in categorical columns."""
        for col, profile in self.profile.get('columns', {}).items():
            col_type = profile.get('type_info', {}).get('type')
            if col_type not in [DataTypeDetector.CATEGORICAL_NOMINAL, DataTypeDetector.CATEGORICAL_ORDINAL]:
                continue
            
            stats = profile.get('statistics', {})
            balance = stats.get('balance_score', 1.0)
            
            if balance < 0.3:
                self._add_insight(
                    'imbalance',
                    f'Column "{col}" has severe class imbalance (balance score: {balance:.2f})',
                    self.SEVERITY_HIGH,
                    {'column': col, 'balance_score': balance}
                )
            elif balance < 0.5:
                self._add_insight(
                    'imbalance',
                    f'Column "{col}" has moderate class imbalance (balance score: {balance:.2f})',
                    self.SEVERITY_MEDIUM,
                    {'column': col, 'balance_score': balance}
                )
    
    def _check_outliers(self):
        """Check for outliers in numeric columns."""
        outlier_columns = []
        
        for col, profile in self.profile.get('columns', {}).items():
            col_type = profile.get('type_info', {}).get('type')
            if col_type not in [DataTypeDetector.NUMERIC_CONTINUOUS, DataTypeDetector.NUMERIC_DISCRETE]:
                continue
            
            stats = profile.get('statistics', {})
            n_outliers = stats.get('outliers_iqr', 0)
            total = stats.get('count', 1)
            outlier_pct = n_outliers / total * 100 if total > 0 else 0
            
            if outlier_pct > 10:
                outlier_columns.append({
                    'column': col,
                    'outlier_count': n_outliers,
                    'outlier_percentage': outlier_pct
                })
        
        if outlier_columns:
            severe = [o for o in outlier_columns if o['outlier_percentage'] > 20]
            if severe:
                self._add_insight(
                    'outliers',
                    f'Found {len(severe)} columns with >20% outliers',
                    self.SEVERITY_HIGH,
                    {'columns': severe[:3]}
                )
            
            self._add_insight(
                'outliers',
                f'{len(outlier_columns)} numeric columns have significant outliers (>10%)',
                self.SEVERITY_MEDIUM if not severe else self.SEVERITY_LOW,
                {'columns': outlier_columns[:5]}
            )
            
            self._add_suggestion(
                'Review outliers - consider transformation or capping',
                'Outliers can significantly impact model performance',
                'medium',
                'Use log transformation, IQR capping, or robust scalers'
            )
    
    def _check_data_types(self):
        """Check for potential data type issues."""
        # Check numeric columns stored as strings
        for col in self.profiler.df.columns:
            if self.profiler.df[col].dtype == object:
                # Try to detect if it should be numeric
                sample = self.profiler.df[col].dropna().head(100)
                if len(sample) > 0:
                    numeric_count = sample.apply(lambda x: str(x).replace('.', '').replace('-', '').isdigit()).sum()
                    if numeric_count / len(sample) > 0.9:
                        self._add_insight(
                            'data_types',
                            f'Column "{col}" appears to be numeric but stored as text',
                            self.SEVERITY_MEDIUM,
                            {'column': col, 'suggested_type': 'numeric'}
                        )
                        self._add_suggestion(
                            f'Convert "{col}" to numeric type',
                            'Numeric data stored as text wastes memory and prevents analysis',
                            'medium',
                            f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')"
                        )
    
    def _check_high_cardinality(self):
        """Check for high cardinality categorical columns."""
        for col, profile in self.profile.get('columns', {}).items():
            col_type = profile.get('type_info', {}).get('type')
            stats = profile.get('statistics', {})
            cardinality = stats.get('unique', 0)
            
            if col_type == DataTypeDetector.CATEGORICAL_NOMINAL and cardinality > 50:
                self._add_insight(
                    'cardinality',
                    f'Column "{col}" has high cardinality ({cardinality} unique values)',
                    self.SEVERITY_MEDIUM,
                    {'column': col, 'cardinality': cardinality}
                )
                
                if cardinality > 100:
                    self._add_suggestion(
                        f'Consider grouping rare categories in "{col}"',
                        f'{cardinality} categories may cause overfitting and sparse features',
                        'medium',
                        'Group categories with <1% frequency into "Other"'
                    )
    
    def _check_constant_columns(self):
        """Check for constant or near-constant columns."""
        for col, profile in self.profile.get('columns', {}).items():
            stats = profile.get('statistics', {})
            unique = stats.get('unique', 0)
            total = stats.get('count', 1)
            
            if unique == 1:
                self._add_insight(
                    'constant',
                    f'Column "{col}" is constant (only 1 unique value)',
                    self.SEVERITY_HIGH,
                    {'column': col}
                )
                self._add_suggestion(
                    f'Drop constant column "{col}"',
                    'Constant columns provide no information',
                    'high',
                    f"df = df.drop('{col}', axis=1)"
                )
            elif unique == 2 and total > 100:
                # Check if almost constant
                value_counts = self.profiler.df[col].value_counts()
                if value_counts.iloc[0] / total > 0.99:
                    self._add_insight(
                        'constant',
                        f'Column "{col}" is nearly constant (99%+ same value)',
                        self.SEVERITY_MEDIUM,
                        {'column': col, 'dominant_value': str(value_counts.index[0])}
                    )
    
    def get_insights(self) -> List[Dict[str, Any]]:
        """Get all insights."""
        return self.insights
    
    def get_suggestions(self) -> List[Dict[str, Any]]:
        """Get all suggestions sorted by priority."""
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        return sorted(self.suggestions, key=lambda x: priority_order.get(x['priority'], 4))
    
    def get_critical_issues(self) -> List[Dict[str, Any]]:
        """Get only critical and high priority issues."""
        return [i for i in self.insights if i['severity'] in [self.SEVERITY_CRITICAL, self.SEVERITY_HIGH]]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of insights and suggestions."""
        severity_counts = {}
        for insight in self.insights:
            sev = insight['severity']
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        return {
            'total_insights': len(self.insights),
            'total_suggestions': len(self.suggestions),
            'severity_distribution': severity_counts,
            'critical_issues': len([i for i in self.insights if i['severity'] == self.SEVERITY_CRITICAL]),
            'high_priority': len([i for i in self.insights if i['severity'] == self.SEVERITY_HIGH]),
        }
