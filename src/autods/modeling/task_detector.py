"""Machine Learning task type detection."""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from ..analysis.type_detector import DataTypeDetector


class MLTaskType:
    """ML task type constants."""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    CLUSTERING = "clustering"
    NLP_CLASSIFICATION = "nlp_classification"
    NLP_REGRESSION = "nlp_regression"
    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_REGRESSION = "image_regression"
    TIME_SERIES = "time_series"
    UNKNOWN = "unknown"


class TaskDetector:
    """Detect the type of ML task based on data characteristics."""
    
    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        self.df = df
        self.target_column = target_column
        self.type_detector = DataTypeDetector(df)
        self.task_info: Dict[str, Any] = {}
        self._detect_task()
    
    def _detect_task(self):
        """Main task detection logic."""
        # Check if we have image data
        if self._is_image_task():
            return
        
        # Check if we have text/NLP task
        if self._is_nlp_task():
            return
        
        # Check if we have time series
        if self._is_time_series():
            return
        
        # Check if we have a target (supervised) or not (unsupervised)
        if self.target_column and self.target_column in self.df.columns:
            self._detect_supervised_task()
        else:
            self._detect_unsupervised_task()
    
    def _is_image_task(self) -> bool:
        """Detect if this is an image task."""
        # Check for image columns
        image_cols = []
        for col in self.df.columns:
            col_info = self.type_detector.columns_info.get(col, {})
            if col_info.get('type') == 'image':
                image_cols.append(col)
        
        if image_cols:
            # Determine if classification or regression based on target
            if self.target_column and self.target_column in self.df.columns:
                target_info = self.type_detector.columns_info.get(self.target_column, {})
                target_type = target_info.get('type')
                
                if target_type in ['categorical_nominal', 'boolean', 'categorical_ordinal']:
                    unique_count = self.df[self.target_column].nunique()
                    if unique_count == 2:
                        self.task_info = {
                            'task_type': MLTaskType.IMAGE_CLASSIFICATION,
                            'subtype': 'binary',
                            'image_columns': image_cols,
                            'description': 'Binary image classification'
                        }
                    else:
                        self.task_info = {
                            'task_type': MLTaskType.IMAGE_CLASSIFICATION,
                            'subtype': 'multiclass',
                            'n_classes': unique_count,
                            'image_columns': image_cols,
                            'description': f'Multiclass image classification ({unique_count} classes)'
                        }
                else:
                    self.task_info = {
                        'task_type': MLTaskType.IMAGE_REGRESSION,
                        'image_columns': image_cols,
                        'description': 'Image regression'
                    }
            else:
                # Unsupervised - default to clustering
                self.task_info = {
                    'task_type': MLTaskType.CLUSTERING,
                    'data_type': 'image',
                    'image_columns': image_cols,
                    'description': 'Image clustering (unsupervised)'
                }
            return True
        
        return False
    
    def _is_nlp_task(self) -> bool:
        """Detect if this is an NLP/text task."""
        # Check for text columns
        text_cols = []
        for col in self.df.columns:
            col_info = self.type_detector.columns_info.get(col, {})
            if col_info.get('type') == 'text':
                text_cols.append(col)
        
        if text_cols:
            # Determine if classification or regression based on target
            if self.target_column and self.target_column in self.df.columns:
                target_info = self.type_detector.columns_info.get(self.target_column, {})
                target_type = target_info.get('type')
                
                if target_type in ['categorical_nominal', 'boolean', 'categorical_ordinal']:
                    unique_count = self.df[self.target_column].nunique()
                    if unique_count == 2:
                        self.task_info = {
                            'task_type': MLTaskType.NLP_CLASSIFICATION,
                            'subtype': 'binary',
                            'text_columns': text_cols,
                            'description': 'Binary text classification'
                        }
                    else:
                        self.task_info = {
                            'task_type': MLTaskType.NLP_CLASSIFICATION,
                            'subtype': 'multiclass',
                            'n_classes': unique_count,
                            'text_columns': text_cols,
                            'description': f'Multiclass text classification ({unique_count} classes)'
                        }
                else:
                    self.task_info = {
                        'task_type': MLTaskType.NLP_REGRESSION,
                        'text_columns': text_cols,
                        'description': 'Text regression'
                    }
            else:
                # Could be clustering or topic modeling
                self.task_info = {
                    'task_type': MLTaskType.CLUSTERING,
                    'data_type': 'text',
                    'text_columns': text_cols,
                    'description': 'Text clustering/topic modeling (unsupervised)'
                }
            return True
        
        return False
    
    def _is_time_series(self) -> bool:
        """Detect if this is a time series task."""
        # Check for datetime columns
        datetime_cols = []
        for col in self.df.columns:
            col_info = self.type_detector.columns_info.get(col, {})
            if col_info.get('type') == 'datetime':
                datetime_cols.append(col)
        
        # Simple heuristic: if there's a datetime and numeric target
        if datetime_cols and self.target_column:
            target_info = self.type_detector.columns_info.get(self.target_column, {})
            if target_info.get('type') in ['numeric_continuous', 'numeric_discrete']:
                self.task_info = {
                    'task_type': MLTaskType.TIME_SERIES,
                    'datetime_columns': datetime_cols,
                    'description': 'Time series forecasting'
                }
                return True
        
        return False
    
    def _detect_supervised_task(self):
        """Detect supervised learning task type."""
        target_info = self.type_detector.columns_info.get(self.target_column, {})
        target_type = target_info.get('type')
        
        if target_type in ['numeric_continuous', 'numeric_discrete']:
            self.task_info = {
                'task_type': MLTaskType.REGRESSION,
                'target_type': 'numeric',
                'description': 'Regression task - predict continuous value'
            }
        elif target_type in ['categorical_nominal', 'boolean']:
            unique_count = self.df[self.target_column].nunique()
            if unique_count == 2:
                self.task_info = {
                    'task_type': MLTaskType.BINARY_CLASSIFICATION,
                    'target_type': 'binary',
                    'description': 'Binary classification task'
                }
            else:
                self.task_info = {
                    'task_type': MLTaskType.MULTICLASS_CLASSIFICATION,
                    'target_type': 'multiclass',
                    'n_classes': unique_count,
                    'description': f'Multiclass classification task ({unique_count} classes)'
                }
        elif target_type == 'categorical_ordinal':
            unique_count = self.df[self.target_column].nunique()
            self.task_info = {
                'task_type': MLTaskType.MULTICLASS_CLASSIFICATION,
                'target_type': 'ordinal',
                'n_classes': unique_count,
                'description': f'Ordinal classification task ({unique_count} classes)'
            }
        else:
            self.task_info = {
                'task_type': MLTaskType.UNKNOWN,
                'target_type': target_type,
                'description': f'Unknown task type with target: {target_type}'
            }
    
    def _detect_unsupervised_task(self):
        """Detect unsupervised learning task type."""
        # Default to clustering for tabular data without target
        numeric_cols = [c for c in self.df.columns 
                       if self.type_detector.columns_info.get(c, {}).get('type') 
                       in ['numeric_continuous', 'numeric_discrete']]
        
        if len(numeric_cols) >= 2:
            self.task_info = {
                'task_type': MLTaskType.CLUSTERING,
                'data_type': 'tabular',
                'description': 'Clustering task - find natural groupings in data',
                'suggested_k': min(10, len(self.df) // 10)  # Simple heuristic
            }
        else:
            self.task_info = {
                'task_type': MLTaskType.UNKNOWN,
                'data_type': 'unknown',
                'description': 'Unable to determine task type - insufficient numeric features for clustering'
            }
    
    def get_task_type(self) -> str:
        """Get detected task type."""
        return self.task_info.get('task_type', MLTaskType.UNKNOWN)
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get full task information."""
        return self.task_info
    
    def is_classification(self) -> bool:
        """Check if task is classification."""
        return self.task_info.get('task_type') in [
            MLTaskType.BINARY_CLASSIFICATION,
            MLTaskType.MULTICLASS_CLASSIFICATION,
            MLTaskType.NLP_CLASSIFICATION,
            MLTaskType.IMAGE_CLASSIFICATION
        ]
    
    def is_regression(self) -> bool:
        """Check if task is regression."""
        return self.task_info.get('task_type') in [
            MLTaskType.REGRESSION,
            MLTaskType.NLP_REGRESSION,
            MLTaskType.IMAGE_REGRESSION
        ]
    
    def is_clustering(self) -> bool:
        """Check if task is clustering."""
        return self.task_info.get('task_type') == MLTaskType.CLUSTERING
    
    def is_supervised(self) -> bool:
        """Check if task is supervised."""
        return self.target_column is not None and self.target_column in self.df.columns
