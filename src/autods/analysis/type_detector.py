"""Data type detection and classification."""
import re
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


class DataTypeDetector:
    """Automatically detect and classify data types in datasets."""
    
    # Type classifications
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    DATETIME = "datetime"
    TEXT = "text"
    ID = "id"
    BOOLEAN = "boolean"
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns_info: Dict[str, Dict[str, Any]] = {}
        self._analyze_all_columns()
    
    def _analyze_all_columns(self):
        """Analyze all columns and store type information."""
        for col in self.df.columns:
            self.columns_info[col] = self._analyze_column(col)
    
    def _analyze_column(self, col: str) -> Dict[str, Any]:
        """Analyze a single column and determine its type."""
        series = self.df[col]
        info = {
            'column': col,
            'dtype': str(series.dtype),
            'null_count': series.isnull().sum(),
            'null_pct': series.isnull().sum() / len(series) * 100,
            'unique_count': series.nunique(dropna=True),
            'total_count': len(series),
        }
        
        # Detect if it's an ID column
        if self._is_id_column(series, col):
            info['type'] = self.ID
            info['subtype'] = 'uuid' if self._is_uuid(series) else 'sequential'
            return info
        
        # Check for datetime
        if self._is_datetime(series):
            info['type'] = self.DATETIME
            info['subtype'] = 'datetime'
            return info
        
        # Check for boolean
        if self._is_boolean(series):
            info['type'] = self.BOOLEAN
            info['subtype'] = 'boolean'
            return info
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(series):
            info.update(self._analyze_numeric(series))
            return info
        
        # Text/Categorical columns
        info.update(self._analyze_text_or_categorical(series))
        return info
    
    def _is_id_column(self, series: pd.Series, col_name: str) -> bool:
        """Detect if column is likely an ID."""
        # Check column name patterns
        id_patterns = ['id', 'pk', 'uuid', 'guid', 'key', 'index', 'identifier']
        if any(pattern in col_name.lower() for pattern in id_patterns):
            # Verify uniqueness
            if series.nunique(dropna=True) / len(series) > 0.95:
                return True
        
        # Check if all values are unique (potential ID)
        if series.nunique(dropna=True) == series.notna().sum():
            # High uniqueness ratio
            if series.nunique() / len(series) > 0.95:
                return True
        
        # UUID pattern detection
        if series.dtype == object and len(series) > 0:
            sample = series.dropna().astype(str).iloc[:100]
            uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            matches = sum(1 for val in sample if re.match(uuid_pattern, val, re.I))
            if matches / len(sample) > 0.8:
                return True
        
        return False
    
    def _is_uuid(self, series: pd.Series) -> bool:
        """Check if values are UUID format."""
        if series.dtype != object:
            return False
        sample = series.dropna().astype(str).iloc[:50]
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        matches = sum(1 for val in sample if re.match(uuid_pattern, val, re.I))
        return matches / len(sample) > 0.8 if len(sample) > 0 else False
    
    def _is_datetime(self, series: pd.Series) -> bool:
        """Detect if column contains datetime values."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        if series.dtype == object:
            # Try parsing sample
            try:
                sample = series.dropna().iloc[:100]
                if len(sample) > 0:
                    pd.to_datetime(sample, infer_datetime_format=True, errors='raise')
                    return True
            except (ValueError, TypeError):
                pass
        
        return False
    
    def _is_boolean(self, series: pd.Series) -> bool:
        """Detect if column is boolean."""
        if pd.api.types.is_bool_dtype(series):
            return True
        
        if series.dtype == object or series.dtype == 'int64':
            unique_vals = set(series.dropna().unique())
            bool_sets = [
                {True, False},
                {0, 1},
                {'True', 'False'},
                {'true', 'false'},
                {'yes', 'no'},
                {'Yes', 'No'},
                {'Y', 'N'},
                {'y', 'n'},
            ]
            if unique_vals in [set(s) for s in bool_sets] or unique_vals.issubset({0, 1}):
                return True
        
        return False
    
    def _analyze_numeric(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numeric column."""
        info = {}
        
        # Calculate statistics
        non_null = series.dropna()
        info['min'] = non_null.min()
        info['max'] = non_null.max()
        info['mean'] = non_null.mean()
        info['median'] = non_null.median()
        info['std'] = non_null.std()
        info['skewness'] = non_null.skew()
        info['kurtosis'] = non_null.kurtosis()
        
        # Determine if continuous or discrete
        unique_ratio = series.nunique() / len(series)
        
        # Check for integers with low unique values (likely categorical/ordinal)
        if pd.api.types.is_integer_dtype(series) and series.nunique() <= 20:
            info['type'] = self.CATEGORICAL_ORDINAL
            info['subtype'] = 'ordinal_numeric'
        elif unique_ratio < 0.05 and series.nunique() <= 50:
            # Low cardinality numeric - might be ordinal encoded
            info['type'] = self.CATEGORICAL_ORDINAL
            info['subtype'] = 'ordinal_encoded'
        else:
            # Continuous numeric
            info['type'] = self.NUMERIC_CONTINUOUS
            info['subtype'] = 'continuous'
        
        info['cardinality'] = series.nunique()
        
        return info
    
    def _analyze_text_or_categorical(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze text or categorical column."""
        info = {}
        non_null = series.dropna().astype(str)
        
        # Calculate text statistics
        info['avg_length'] = non_null.str.len().mean() if len(non_null) > 0 else 0
        info['max_length'] = non_null.str.len().max() if len(non_null) > 0 else 0
        info['min_length'] = non_null.str.len().min() if len(non_null) > 0 else 0
        
        cardinality = series.nunique()
        cardinality_ratio = cardinality / len(series)
        
        # Determine categorical vs text
        if cardinality <= 20:
            # Low cardinality - categorical
            info['type'] = self.CATEGORICAL_NOMINAL
            info['subtype'] = 'low_cardinality'
            info['categories'] = series.dropna().unique().tolist()[:20]
        elif cardinality <= 100 and cardinality_ratio < 0.1:
            # Medium cardinality - likely categorical
            info['type'] = self.CATEGORICAL_NOMINAL
            info['subtype'] = 'medium_cardinality'
        elif cardinality_ratio > 0.5 or (info['avg_length'] > 50 and cardinality_ratio > 0.3):
            # High uniqueness and/or long text - free text
            info['type'] = self.TEXT
            info['subtype'] = 'free_text'
        else:
            # Default to categorical
            info['type'] = self.CATEGORICAL_NOMINAL
            info['subtype'] = 'categorical'
        
        info['cardinality'] = cardinality
        
        return info
    
    def get_column_types(self) -> Dict[str, str]:
        """Get mapping of column names to types."""
        return {col: info['type'] for col, info in self.columns_info.items()}
    
    def get_columns_by_type(self, data_type: str) -> List[str]:
        """Get all columns of a specific type."""
        return [col for col, info in self.columns_info.items() if info['type'] == data_type]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all column types."""
        type_counts = {}
        for info in self.columns_info.values():
            t = info['type']
            type_counts[t] = type_counts.get(t, 0) + 1
        
        return {
            'total_columns': len(self.columns_info),
            'type_distribution': type_counts,
            'columns': self.columns_info
        }
    
    def suggest_target_column(self) -> Optional[str]:
        """Suggest the most likely target column."""
        candidates = []
        
        for col, info in self.columns_info.items():
            # Common target patterns in column name
            target_indicators = ['target', 'label', 'y', 'class', 'category', 'output', 'prediction']
            name_score = sum(1 for ind in target_indicators if ind in col.lower())
            
            # Prefer columns with:
            # - Binary classification: 2 unique values
            # - Multi-class: 2-20 unique values
            # - Regression: continuous numeric
            
            if info['type'] in [self.CATEGORICAL_NOMINAL, self.CATEGORICAL_ORDINAL, self.BOOLEAN]:
                if 2 <= info['unique_count'] <= 100:
                    candidates.append((col, name_score + 2, info['unique_count']))
            elif info['type'] == self.NUMERIC_CONTINUOUS:
                # Regression target
                candidates.append((col, name_score + 1, 0))
        
        # Sort by score (descending) and return best candidate
        if candidates:
            candidates.sort(key=lambda x: (-x[1], -x[2]))
            return candidates[0][0]
        
        return None
