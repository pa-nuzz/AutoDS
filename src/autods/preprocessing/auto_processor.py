"""Auto-preprocessing pipeline - 'Do it for me' mode."""
import pickle
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    OneHotEncoder, OrdinalEncoder, LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from .detector import PreprocessingDetector, PreprocessingNeed, NeedSeverity


class AutoPreprocessor:
    """Automatically preprocess data based on detected needs."""
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 target_column: Optional[str] = None,
                 mode: str = "balanced"):  # "minimal", "balanced", "aggressive"
        self.df_original = df.copy()
        self.target_column = target_column
        self.mode = mode
        self.df_processed = None
        self.pipeline = None
        self.pipeline_steps = []
        self.transformers = {}
        
        # Detect needs
        self.detector = PreprocessingDetector(df, target_column)
        self.needs = self.detector.get_needs()
        
        # Categorize columns
        self.numeric_cols = []
        self.categorical_cols = []
        self.text_cols = []
        self.datetime_cols = []
        self._categorize_columns()
    
    def _categorize_columns(self):
        """Categorize columns by type."""
        for col in self.df_original.columns:
            if col == self.target_column:
                continue
            
            col_info = self.detector.type_detector.columns_info.get(col, {})
            col_type = col_info.get('type')
            
            if col_type in ['numeric_continuous', 'numeric_discrete']:
                self.numeric_cols.append(col)
            elif col_type in ['categorical_nominal', 'categorical_ordinal', 'boolean']:
                self.categorical_cols.append(col)
            elif col_type == 'text':
                self.text_cols.append(col)
            elif col_type == 'datetime':
                self.datetime_cols.append(col)
    
    def fit_transform(self) -> pd.DataFrame:
        """Fit and apply all preprocessing steps."""
        df = self.df_original.copy()
        
        # 1. Remove duplicates
        df = self._handle_duplicates(df)
        
        # 2. Handle datetime features
        df = self._extract_datetime_features(df)
        
        # 3. Handle missing values
        df = self._handle_missing_values(df)
        
        # 4. Handle outliers (if aggressive mode)
        if self.mode == "aggressive":
            df = self._handle_outliers(df)
        
        # 5. Encode categorical variables
        df = self._encode_categorical(df)
        
        # 6. Scale numeric features
        df = self._scale_numeric(df)
        
        # 7. Handle text features
        if self.text_cols:
            df = self._handle_text(df)
        
        self.df_processed = df
        return df
    
    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        dup_needs = self.detector.get_needs_by_type(PreprocessingNeed.DUPLICATE_REMOVAL)
        if dup_needs:
            initial_rows = len(df)
            df = df.drop_duplicates()
            removed = initial_rows - len(df)
            self.pipeline_steps.append(f"Removed {removed} duplicate rows")
        return df
    
    def _extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from datetime columns."""
        for col in self.datetime_cols:
            if col not in df.columns:
                continue
            
            try:
                dt_series = pd.to_datetime(df[col], errors='coerce')
                
                # Extract components
                df[f'{col}_year'] = dt_series.dt.year
                df[f'{col}_month'] = dt_series.dt.month
                df[f'{col}_day'] = dt_series.dt.day
                df[f'{col}_dayofweek'] = dt_series.dt.dayofweek
                df[f'{col}_quarter'] = dt_series.dt.quarter
                
                # Drop original datetime column
                df = df.drop(columns=[col])
                self.pipeline_steps.append(f"Extracted datetime features from {col}")
            except Exception:
                pass
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on column type and severity."""
        missing_needs = self.detector.get_needs_by_type(PreprocessingNeed.MISSING_VALUES)
        
        for need in missing_needs:
            cols = need['columns']
            strategy = need['details'].get('strategy', 'simple_imputation')
            
            for col in cols:
                if col not in df.columns:
                    continue
                
                if strategy == 'drop_column':
                    # Drop columns with >50% missing
                    df = df.drop(columns=[col])
                    self.pipeline_steps.append(f"Dropped column '{col}' (>50% missing)")
                    if col in self.numeric_cols:
                        self.numeric_cols.remove(col)
                    if col in self.categorical_cols:
                        self.categorical_cols.remove(col)
                
                elif strategy in ['simple_imputation', 'advanced_imputation']:
                    # Determine imputation strategy based on column type
                    if col in self.numeric_cols:
                        # Use median for numeric (robust to outliers)
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val)
                        self.pipeline_steps.append(f"Imputed '{col}' with median ({median_val:.2f})")
                    else:
                        # Use mode for categorical
                        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                        df[col] = df[col].fillna(mode_val)
                        self.pipeline_steps.append(f"Imputed '{col}' with mode ({mode_val})")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers in numeric columns."""
        outlier_needs = self.detector.get_needs_by_type(PreprocessingNeed.OUTLIER_TREATMENT)
        
        for need in outlier_needs:
            for col in need['columns']:
                if col not in df.columns or col not in self.numeric_cols:
                    continue
                
                # IQR-based capping
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_before = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                if outliers_before > 0:
                    self.pipeline_steps.append(f"Capped outliers in '{col}' ({outliers_before} values)")
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        encoding_needs = self.detector.get_needs_by_type(PreprocessingNeed.ENCODING)
        
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            
            unique_count = df[col].nunique()
            
            # Choose encoding strategy based on cardinality
            if unique_count <= 10:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                self.pipeline_steps.append(f"One-hot encoded '{col}' ({unique_count} categories)")
            elif unique_count <= 50:
                # One-hot with drop_first to avoid multicollinearity
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                self.pipeline_steps.append(f"One-hot encoded '{col}' with drop_first")
            else:
                # Ordinal encoding for high cardinality
                mapping = {cat: i for i, cat in enumerate(df[col].unique())}
                df[col] = df[col].map(mapping)
                self.transformers[f'{col}_mapping'] = mapping
                self.pipeline_steps.append(f"Ordinal encoded '{col}' ({unique_count} categories)")
        
        return df
    
    def _scale_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric features."""
        if not self.numeric_cols:
            return df
        
        # Filter only existing numeric columns
        existing_numeric = [col for col in self.numeric_cols if col in df.columns]
        if not existing_numeric:
            return df
        
        # Determine scaler based on mode and outlier presence
        outlier_needs = self.detector.get_needs_by_type(PreprocessingNeed.OUTLIER_TREATMENT)
        cols_with_outliers = set()
        for need in outlier_needs:
            cols_with_outliers.update(need['columns'])
        
        if self.mode == "aggressive" or existing_numeric and any(col in cols_with_outliers for col in existing_numeric):
            scaler = RobustScaler()
            scaler_name = "RobustScaler"
        elif self.mode == "minimal":
            scaler = MinMaxScaler()
            scaler_name = "MinMaxScaler"
        else:
            scaler = StandardScaler()
            scaler_name = "StandardScaler"
        
        # Apply scaling
        df[existing_numeric] = scaler.fit_transform(df[existing_numeric])
        self.transformers['scaler'] = scaler
        self.pipeline_steps.append(f"Applied {scaler_name} to {len(existing_numeric)} numeric columns")
        
        return df
    
    def _handle_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle text features using TF-IDF."""
        text_vectors = []
        
        for col in self.text_cols:
            if col not in df.columns:
                continue
            
            # Fill missing text with empty string
            text_data = df[col].fillna('').astype(str)
            
            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            try:
                vectors = vectorizer.fit_transform(text_data)
                
                # Convert to DataFrame
                feature_names = [f"{col}_tfidf_{i}" for i in range(vectors.shape[1])]
                vector_df = pd.DataFrame(
                    vectors.toarray(),
                    columns=feature_names,
                    index=df.index
                )
                
                text_vectors.append(vector_df)
                self.transformers[f'{col}_vectorizer'] = vectorizer
                self.pipeline_steps.append(f"TF-IDF vectorized '{col}' ({vectors.shape[1]} features)")
            except Exception as e:
                # If vectorization fails, drop the text column
                self.pipeline_steps.append(f"Skipped text vectorization for '{col}': {e}")
        
        # Combine all features
        if text_vectors:
            df = df.drop(columns=[col for col in self.text_cols if col in df.columns])
            df = pd.concat([df] + text_vectors, axis=1)
        
        return df
    
    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """Get processed dataframe."""
        return self.df_processed
    
    def get_feature_names(self) -> List[str]:
        """Get list of processed feature names."""
        if self.df_processed is None:
            return []
        cols = list(self.df_processed.columns)
        if self.target_column and self.target_column in cols:
            cols.remove(self.target_column)
        return cols
    
    def get_summary(self) -> Dict[str, Any]:
        """Get preprocessing summary."""
        return {
            'original_shape': self.df_original.shape,
            'processed_shape': self.df_processed.shape if self.df_processed is not None else None,
            'mode': self.mode,
            'steps_applied': self.pipeline_steps,
            'needs_detected': len(self.needs),
            'needs_required': len([n for n in self.needs if n['severity'] == NeedSeverity.REQUIRED]),
            'numeric_features': len([c for c in self.numeric_cols if c in self.get_feature_names()]),
            'categorical_features': len(self.categorical_cols),
            'text_features': len(self.text_cols),
        }
    
    def save(self, filepath: str):
        """Save preprocessor state."""
        state = {
            'pipeline_steps': self.pipeline_steps,
            'transformers': self.transformers,
            'numeric_cols': self.numeric_cols,
            'categorical_cols': self.categorical_cols,
            'text_cols': self.text_cols,
            'target_column': self.target_column,
            'mode': self.mode,
            'needs': self.needs,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, filepath: str) -> Dict[str, Any]:
        """Load preprocessor state."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class PreprocessingPipeline:
    """Sklearn-compatible preprocessing pipeline."""
    
    def __init__(self, 
                 numeric_strategy: str = "standard",
                 categorical_strategy: str = "onehot",
                 impute_strategy: str = "median"):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.impute_strategy = impute_strategy
        self.pipeline = None
        self._build_pipeline()
    
    def _build_pipeline(self):
        """Build sklearn pipeline."""
        # Numeric transformer
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.impute_strategy)),
            ('scaler', self._get_scaler())
        ])
        
        # Categorical transformer
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', self._get_encoder())
        ])
        
        # Combine
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, []),  # To be filled during fit
                ('cat', categorical_transformer, [])  # To be filled during fit
            ],
            remainder='drop'
        )
    
    def _get_scaler(self):
        """Get scaler based on strategy."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'maxabs': MaxAbsScaler(),
        }
        return scalers.get(self.numeric_strategy, StandardScaler())
    
    def _get_encoder(self):
        """Get encoder based on strategy."""
        encoders = {
            'onehot': OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            'ordinal': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
        }
        return encoders.get(self.categorical_strategy, OneHotEncoder())
    
    def fit_transform(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> np.ndarray:
        """Fit and transform data."""
        self.pipeline.transformers[0] = ('num', self.pipeline.transformers[0][1], numeric_cols)
        self.pipeline.transformers[1] = ('cat', self.pipeline.transformers[1][1], categorical_cols)
        return self.pipeline.fit_transform(df)
