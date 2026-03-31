"""Statistical profiling for datasets."""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats

from .type_detector import DataTypeDetector


class StatisticalProfiler:
    """Generate comprehensive statistical profiles of datasets."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.type_detector = DataTypeDetector(df)
        self.profile: Dict[str, Any] = {}
        self._generate_profile()
    
    def _generate_profile(self):
        """Generate complete profile."""
        self.profile = {
            'overview': self._overview_stats(),
            'columns': self._column_profiles(),
            'correlations': self._correlation_analysis(),
            'missing_values': self._missing_value_analysis(),
            'duplicates': self._duplicate_analysis(),
            'target_suggestion': self.type_detector.suggest_target_column(),
            'quality_score': self._compute_quality_score(),
        }
    
    def _overview_stats(self) -> Dict[str, Any]:
        """Generate dataset overview statistics."""
        return {
            'n_rows': len(self.df),
            'n_columns': len(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / (1024 * 1024),
            'total_cells': len(self.df) * len(self.df.columns),
            'missing_cells': self.df.isnull().sum().sum(),
            'missing_pct': self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100,
            'duplicate_rows': self.df.duplicated().sum(),
            'data_types': self.df.dtypes.apply(str).to_dict(),
        }
    
    def _column_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Generate detailed profiles for each column."""
        profiles = {}
        
        for col in self.df.columns:
            col_info = self.type_detector.columns_info[col]
            col_type = col_info['type']
            series = self.df[col]
            
            profile = {
                'type_info': col_info,
                'statistics': self._compute_statistics(series, col_type),
            }
            
            profiles[col] = profile
        
        return profiles
    
    def _compute_statistics(self, series: pd.Series, col_type: str) -> Dict[str, Any]:
        """Compute statistics based on column type."""
        stats_dict = {
            'count': series.count(),
            'missing': series.isnull().sum(),
            'missing_pct': series.isnull().sum() / len(series) * 100,
            'unique': series.nunique(dropna=True),
        }
        
        if col_type in [DataTypeDetector.NUMERIC_CONTINUOUS, DataTypeDetector.NUMERIC_DISCRETE]:
            non_null = series.dropna()
            if len(non_null) > 0:
                stats_dict.update({
                    'mean': float(non_null.mean()),
                    'median': float(non_null.median()),
                    'std': float(non_null.std()),
                    'min': float(non_null.min()),
                    'max': float(non_null.max()),
                    'range': float(non_null.max() - non_null.min()),
                    'q25': float(non_null.quantile(0.25)),
                    'q75': float(non_null.quantile(0.75)),
                    'iqr': float(non_null.quantile(0.75) - non_null.quantile(0.25)),
                    'skewness': float(non_null.skew()),
                    'kurtosis': float(non_null.kurtosis()),
                    'zeros': (non_null == 0).sum(),
                    'zeros_pct': (non_null == 0).sum() / len(non_null) * 100,
                    'negative': (non_null < 0).sum(),
                    'outliers_iqr': self._count_outliers_iqr(non_null),
                    'outliers_zscore': self._count_outliers_zscore(non_null),
                })
        
        elif col_type in [DataTypeDetector.CATEGORICAL_NOMINAL, DataTypeDetector.CATEGORICAL_ORDINAL, DataTypeDetector.BOOLEAN]:
            stats_dict['top_categories'] = series.value_counts().head(10).to_dict()
            stats_dict['category_distribution'] = (
                series.value_counts(normalize=True).head(10) * 100
            ).to_dict()
            
            # Entropy (measure of imbalance)
            vc = series.value_counts(normalize=True)
            stats_dict['entropy'] = float(-(vc * np.log2(vc + 1e-10)).sum())
            n_cats = len(vc)
            if n_cats > 1:
                balance = 1 - (vc.max() - 1/n_cats) / max(1 - 1/n_cats, 1e-10)
                stats_dict['balance_score'] = float(max(0.0, min(1.0, balance)))
            else:
                stats_dict['balance_score'] = 0.0
        
        elif col_type == DataTypeDetector.TEXT:
            non_null = series.dropna().astype(str)
            if len(non_null) > 0:
                lengths = non_null.str.len()
                stats_dict.update({
                    'avg_length': float(lengths.mean()),
                    'max_length': int(lengths.max()),
                    'min_length': int(lengths.min()),
                    'empty_strings': (non_null == '').sum(),
                    'empty_pct': (non_null == '').sum() / len(non_null) * 100,
                })
        
        elif col_type == DataTypeDetector.DATETIME:
            non_null = pd.to_datetime(series.dropna(), errors='coerce').dropna()
            if len(non_null) > 0:
                stats_dict.update({
                    'min_date': str(non_null.min()),
                    'max_date': str(non_null.max()),
                    'range_days': (non_null.max() - non_null.min()).days,
                })
        
        return stats_dict
    
    def _count_outliers_iqr(self, series: pd.Series) -> int:
        """Count outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return int(((series < lower_bound) | (series > upper_bound)).sum())
    
    def _count_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> int:
        """Count outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
        return int((z_scores > threshold).sum())
    
    def _correlation_analysis(self) -> Dict[str, Any]:
        """Analyze correlations between numeric columns."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {'available': False, 'reason': 'Need at least 2 numeric columns'}
        
        # Compute correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        
        # Find high correlations
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # Threshold for high correlation
                    high_corr.append({
                        'col1': corr_matrix.columns[i],
                        'col2': corr_matrix.columns[j],
                        'correlation': float(corr_val),
                        'strength': 'strong' if abs(corr_val) > 0.8 else 'moderate'
                    })
        
        return {
            'available': True,
            'numeric_columns': numeric_cols,
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': sorted(high_corr, key=lambda x: abs(x['correlation']), reverse=True),
            'n_high_corr': len(high_corr),
        }
    
    def _missing_value_analysis(self) -> Dict[str, Any]:
        """Analyze missing values patterns."""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        missing_by_col = {
            col: {
                'count': int(missing[col]),
                'percentage': float(missing_pct[col]),
                'severity': 'critical' if missing_pct[col] > 50 else 'high' if missing_pct[col] > 20 else 'moderate' if missing_pct[col] > 5 else 'low'
            }
            for col in missing.index if missing[col] > 0
        }
        
        # Columns with no missing values
        complete_cols = [col for col in self.df.columns if missing[col] == 0]
        
        # Missing patterns
        rows_with_missing = self.df.isnull().any(axis=1).sum()
        
        return {
            'total_missing': int(missing.sum()),
            'missing_percentage': float(missing.sum() / (len(self.df) * len(self.df.columns)) * 100),
            'columns_with_missing': len(missing_by_col),
            'complete_columns': len(complete_cols),
            'rows_with_missing': int(rows_with_missing),
            'rows_complete': int(len(self.df) - rows_with_missing),
            'missing_by_column': missing_by_col,
            'complete_column_list': complete_cols,
        }
    
    def _duplicate_analysis(self) -> Dict[str, Any]:
        """Analyze duplicate rows."""
        duplicates = self.df.duplicated().sum()
        
        return {
            'total_duplicates': int(duplicates),
            'duplicate_percentage': float(duplicates / len(self.df) * 100) if len(self.df) > 0 else 0,
            'is_critical': duplicates / len(self.df) > 0.1 if len(self.df) > 0 else False,
        }
    
    def _compute_quality_score(self) -> Dict[str, float]:
        """Compute overall data quality score."""
        n = len(self.df)
        if n == 0:
            return {'overall': 0.0, 'completeness': 0.0, 'uniqueness': 0.0, 'consistency': 0.0}
        
        # Completeness (1 - missing percentage)
        missing_pct = self.df.isnull().sum().sum() / (n * len(self.df.columns))
        completeness = max(0, 1 - missing_pct)
        
        # Uniqueness (1 - duplicate percentage)
        duplicates_pct = self.df.duplicated().sum() / n
        uniqueness = max(0, 1 - duplicates_pct)
        
        # Consistency (based on type consistency)
        consistent_cols = sum(
            1 for col in self.df.columns 
            if self.df[col].dropna().apply(type).nunique() <= 1
        )
        consistency = consistent_cols / len(self.df.columns)
        
        # Overall score (weighted average)
        overall = (completeness * 0.4 + uniqueness * 0.3 + consistency * 0.3)
        
        return {
            'overall': round(overall * 100, 1),
            'completeness': round(completeness * 100, 1),
            'uniqueness': round(uniqueness * 100, 1),
            'consistency': round(consistency * 100, 1),
        }
    
    def get_profile(self) -> Dict[str, Any]:
        """Get complete profile."""
        return self.profile
    
    def get_column_profile(self, column: str) -> Optional[Dict[str, Any]]:
        """Get profile for specific column."""
        return self.profile.get('columns', {}).get(column)
