"""Preprocessing need detector."""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..analysis.type_detector import DataTypeDetector
from ..analysis.profiler import StatisticalProfiler


class PreprocessingNeed:
    """Classification of preprocessing needs."""
    MISSING_VALUES = "missing_values"
    ENCODING = "encoding"
    SCALING = "scaling"
    OUTLIER_TREATMENT = "outlier_treatment"
    TEXT_VECTORIZATION = "text_vectorization"
    IMAGE_NORMALIZATION = "image_normalization"
    DUPLICATE_REMOVAL = "duplicate_removal"
    FEATURE_SELECTION = "feature_selection"


class NeedSeverity:
    """Severity levels for preprocessing needs."""
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class PreprocessingDetector:
    """Detect preprocessing needs for a dataset."""
    
    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        self.df = df.copy()
        self.target_column = target_column
        self.type_detector = DataTypeDetector(df)
        self.profiler = StatisticalProfiler(df)
        self.needs: List[Dict[str, Any]] = []
        self._detect_all()
    
    def _detect_all(self):
        """Detect all preprocessing needs."""
        self._detect_missing_values()
        self._detect_encoding_needs()
        self._detect_scaling_needs()
        self._detect_outliers()
        self._detect_text_needs()
        self._detect_duplicates()
        self._detect_high_cardinality()
        self._detect_class_imbalance()
    
    def _add_need(self, need_type: str, severity: str, columns: List[str],
                  description: str, suggested_action: str, details: Optional[Dict] = None):
        """Add a preprocessing need."""
        self.needs.append({
            'type': need_type,
            'severity': severity,
            'columns': columns,
            'description': description,
            'suggested_action': suggested_action,
            'details': details or {}
        })
    
    def _detect_missing_values(self):
        """Detect missing value issues."""
        missing_analysis = self.profiler.profile.get('missing_values', {})
        
        if missing_analysis.get('total_missing', 0) == 0:
            return
        
        # Critical: >50% missing
        critical_cols = [
            col for col, info in missing_analysis.get('missing_by_column', {}).items()
            if info['severity'] == 'critical'
        ]
        if critical_cols:
            self._add_need(
                PreprocessingNeed.MISSING_VALUES,
                NeedSeverity.REQUIRED,
                critical_cols,
                f"{len(critical_cols)} columns have >50% missing values",
                "Consider dropping these columns",
                {'threshold': '>50%', 'strategy': 'drop_column'}
            )
        
        # High: 20-50% missing
        high_cols = [
            col for col, info in missing_analysis.get('missing_by_column', {}).items()
            if info['severity'] == 'high'
        ]
        if high_cols:
            self._add_need(
                PreprocessingNeed.MISSING_VALUES,
                NeedSeverity.REQUIRED,
                high_cols,
                f"{len(high_cols)} columns have 20-50% missing values",
                "Use advanced imputation (KNN, IterativeImputer)",
                {'threshold': '20-50%', 'strategy': 'advanced_imputation'}
            )
        
        # Moderate: 5-20% missing
        moderate_cols = [
            col for col, info in missing_analysis.get('missing_by_column', {}).items()
            if info['severity'] == 'moderate'
        ]
        if moderate_cols:
            self._add_need(
                PreprocessingNeed.MISSING_VALUES,
                NeedSeverity.RECOMMENDED,
                moderate_cols,
                f"{len(moderate_cols)} columns have 5-20% missing values",
                "Use simple imputation (mean/median/mode)",
                {'threshold': '5-20%', 'strategy': 'simple_imputation'}
            )
    
    def _detect_encoding_needs(self):
        """Detect categorical encoding needs."""
        categorical_cols = []
        ordinal_cols = []
        
        for col in self.df.columns:
            if col == self.target_column:
                continue
            
            col_info = self.type_detector.columns_info.get(col, {})
            col_type = col_info.get('type')
            
            if col_type == DataTypeDetector.CATEGORICAL_NOMINAL:
                categorical_cols.append(col)
            elif col_type == DataTypeDetector.CATEGORICAL_ORDINAL:
                ordinal_cols.append(col)
        
        if categorical_cols:
            self._add_need(
                PreprocessingNeed.ENCODING,
                NeedSeverity.REQUIRED,
                categorical_cols,
                f"{len(categorical_cols)} nominal categorical columns need encoding",
                "Use OneHotEncoder (low cardinality) or TargetEncoder (high cardinality)",
                {'encoding_type': 'nominal', 'methods': ['one_hot', 'target']}
            )
        
        if ordinal_cols:
            self._add_need(
                PreprocessingNeed.ENCODING,
                NeedSeverity.REQUIRED,
                ordinal_cols,
                f"{len(ordinal_cols)} ordinal categorical columns need encoding",
                "Use OrdinalEncoder preserving order",
                {'encoding_type': 'ordinal', 'methods': ['ordinal']}
            )
    
    def _detect_scaling_needs(self):
        """Detect numeric scaling needs."""
        numeric_cols = []
        needs_robust = []
        
        for col in self.df.columns:
            if col == self.target_column:
                continue
            
            col_info = self.type_detector.columns_info.get(col, {})
            col_type = col_info.get('type')
            
            if col_type in [DataTypeDetector.NUMERIC_CONTINUOUS, DataTypeDetector.NUMERIC_DISCRETE]:
                numeric_cols.append(col)
                
                # Check if needs robust scaling (outliers present)
                stats = self.profiler.profile.get('columns', {}).get(col, {}).get('statistics', {})
                if stats.get('outliers_iqr', 0) > 0:
                    needs_robust.append(col)
        
        if numeric_cols:
            if needs_robust:
                self._add_need(
                    PreprocessingNeed.SCALING,
                    NeedSeverity.RECOMMENDED,
                    needs_robust,
                    f"{len(needs_robust)} numeric columns have outliers - use robust scaling",
                    "Use RobustScaler to handle outliers",
                    {'scaler': 'robust', 'reason': 'outliers_detected', 'all_numeric': numeric_cols}
                )
            else:
                self._add_need(
                    PreprocessingNeed.SCALING,
                    NeedSeverity.RECOMMENDED,
                    numeric_cols,
                    f"{len(numeric_cols)} numeric columns should be scaled",
                    "Use StandardScaler or MinMaxScaler",
                    {'scaler': 'standard', 'all_numeric': numeric_cols}
                )
    
    def _detect_outliers(self):
        """Detect outlier treatment needs."""
        outlier_cols = []
        
        for col in self.df.columns:
            col_info = self.type_detector.columns_info.get(col, {})
            col_type = col_info.get('type')
            
            if col_type not in [DataTypeDetector.NUMERIC_CONTINUOUS]:
                continue
            
            stats = self.profiler.profile.get('columns', {}).get(col, {}).get('statistics', {})
            outlier_pct = stats.get('outliers_iqr', 0) / stats.get('count', 1) * 100
            
            if outlier_pct > 10:
                outlier_cols.append((col, outlier_pct))
        
        if outlier_cols:
            cols = [c[0] for c in outlier_cols]
            self._add_need(
                PreprocessingNeed.OUTLIER_TREATMENT,
                NeedSeverity.RECOMMENDED,
                cols,
                f"{len(outlier_cols)} columns have >10% outliers",
                "Consider capping, log transformation, or removing outliers",
                {'columns_with_pct': outlier_cols}
            )
    
    def _detect_text_needs(self):
        """Detect text preprocessing needs."""
        text_cols = []
        
        for col in self.df.columns:
            col_info = self.type_detector.columns_info.get(col, {})
            col_type = col_info.get('type')
            
            if col_type == DataTypeDetector.TEXT:
                text_cols.append(col)
        
        if text_cols:
            self._add_need(
                PreprocessingNeed.TEXT_VECTORIZATION,
                NeedSeverity.RECOMMENDED,
                text_cols,
                f"{len(text_cols)} text columns need vectorization",
                "Use TF-IDF or embeddings (Word2Vec, BERT)",
                {'methods': ['tfidf', 'embeddings']}
            )
    
    def _detect_duplicates(self):
        """Detect duplicate removal needs."""
        dup_analysis = self.profiler.profile.get('duplicates', {})
        dup_pct = dup_analysis.get('duplicate_percentage', 0)
        
        if dup_pct > 1:
            severity = NeedSeverity.REQUIRED if dup_pct > 10 else NeedSeverity.RECOMMENDED
            self._add_need(
                PreprocessingNeed.DUPLICATE_REMOVAL,
                severity,
                ['__all__'],
                f"{dup_analysis.get('total_duplicates', 0)} duplicate rows ({dup_pct:.1f}%)",
                "Remove duplicate rows",
                {'duplicate_count': dup_analysis.get('total_duplicates', 0), 'percentage': dup_pct}
            )
    
    def _detect_high_cardinality(self):
        """Detect high cardinality categorical columns."""
        high_card_cols = []
        
        for col in self.df.columns:
            if col == self.target_column:
                continue
            
            col_info = self.type_detector.columns_info.get(col, {})
            col_type = col_info.get('type')
            
            if col_type != DataTypeDetector.CATEGORICAL_NOMINAL:
                continue
            
            unique_count = self.df[col].nunique()
            if unique_count > 50:
                high_card_cols.append((col, unique_count))
        
        if high_card_cols:
            cols = [c[0] for c in high_card_cols]
            self._add_need(
                PreprocessingNeed.FEATURE_SELECTION,
                NeedSeverity.OPTIONAL,
                cols,
                f"{len(high_card_cols)} columns have high cardinality (>50 categories)",
                "Group rare categories or use target encoding",
                {'columns_with_counts': high_card_cols}
            )
    
    def _detect_class_imbalance(self):
        """Detect if target has class imbalance."""
        if not self.target_column:
            return
        
        target_series = self.df[self.target_column]
        col_info = self.type_detector.columns_info.get(self.target_column, {})
        col_type = col_info.get('type')
        
        if col_type not in [DataTypeDetector.CATEGORICAL_NOMINAL, DataTypeDetector.BOOLEAN]:
            return
        
        value_counts = target_series.value_counts(normalize=True)
        if len(value_counts) > 1:
            imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[1]
            if imbalance_ratio > 3:
                self._add_need(
                    'class_imbalance',
                    NeedSeverity.RECOMMENDED,
                    [self.target_column],
                    f"Target has class imbalance (ratio {imbalance_ratio:.1f}:1)",
                    "Use SMOTE, class weights, or stratified sampling",
                    {'imbalance_ratio': imbalance_ratio, 'class_distribution': value_counts.to_dict()}
                )
    
    def get_needs(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all detected needs, optionally filtered by severity."""
        if severity:
            return [n for n in self.needs if n['severity'] == severity]
        return self.needs
    
    def get_needs_by_type(self, need_type: str) -> List[Dict[str, Any]]:
        """Get needs by type."""
        return [n for n in self.needs if n['type'] == need_type]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing needs."""
        required = len([n for n in self.needs if n['severity'] == NeedSeverity.REQUIRED])
        recommended = len([n for n in self.needs if n['severity'] == NeedSeverity.RECOMMENDED])
        optional = len([n for n in self.needs if n['severity'] == NeedSeverity.OPTIONAL])
        
        by_type = {}
        for need in self.needs:
            t = need['type']
            by_type[t] = by_type.get(t, 0) + 1
        
        return {
            'total_needs': len(self.needs),
            'required': required,
            'recommended': recommended,
            'optional': optional,
            'by_type': by_type,
            'can_auto_process': all(n['severity'] != 'required' or n['type'] in [
                PreprocessingNeed.MISSING_VALUES,
                PreprocessingNeed.ENCODING,
                PreprocessingNeed.SCALING,
                PreprocessingNeed.DUPLICATE_REMOVAL
            ] for n in self.needs)
        }
