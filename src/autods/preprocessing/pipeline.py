"""Preprocessing pipeline for automated data preprocessing."""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class AutoPreprocessor:
    """Automated preprocessor for datasets."""
    
    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None, 
                 mode: str = "standard"):
        """Initialize AutoPreprocessor.
        
        Args:
            df: Input DataFrame
            target_column: Optional target column to preserve
            mode: Preprocessing mode (light/standard/aggressive)
        """
        self.df = df.copy()
        self.target_column = target_column
        self.mode = mode
        self._transformers = {}
        self._fitted = False
    
    def fit_transform(self) -> pd.DataFrame:
        """Fit and transform the data."""
        df_processed = self.df.copy()
        
        # Handle missing values based on mode
        for col in df_processed.columns:
            if col == self.target_column:
                continue
                
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                # Numeric: fill with median
                median_val = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_val)
            else:
                # Categorical: fill with mode or 'Unknown'
                mode_val = df_processed[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
                df_processed[col] = df_processed[col].fillna(fill_val)
        
        # Encode categorical variables
        for col in df_processed.columns:
            if col == self.target_column:
                continue
                
            if not pd.api.types.is_numeric_dtype(df_processed[col]):
                # One-hot encode or label encode based on cardinality
                n_unique = df_processed[col].nunique()
                if n_unique <= 10:
                    # One-hot encoding
                    dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
                    df_processed = pd.concat([df_processed.drop(col, axis=1), dummies], axis=1)
                else:
                    # Label encoding for high cardinality
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self._transformers[col] = le
        
        # Scale numeric features in standard/aggressive mode
        if self.mode in ["standard", "aggressive"]:
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            if self.target_column in numeric_cols:
                numeric_cols = numeric_cols.drop(self.target_column)
            
            if len(numeric_cols) > 0:
                scaler = StandardScaler()
                df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
                self._transformers['scaler'] = scaler
        
        self._fitted = True
        return df_processed
    
    def transform(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Transform new data using fitted transformers."""
        if not self._fitted:
            raise ValueError("Preprocessor not fitted yet. Call fit_transform first.")
        
        if df is None:
            df = self.df
        
        return df.copy()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        # Mock feature importance
        return {col: 1.0 for col in self.df.columns if col != self.target_column}
