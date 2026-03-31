"""Data profile dataclass and utilities."""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


@dataclass
class DataProfile:
    """Data profile for a dataset."""
    
    n_rows: int
    n_columns: int
    column_types: Dict[str, str]
    statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    insights: List[Dict[str, Any]] = field(default_factory=list)
    target_suggestion: Optional[str] = None
    quality_score: float = 0.0
    correlations: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        result = {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "column_types": self.column_types,
            "statistics": self.statistics,
            "insights": self.insights,
            "target_suggestion": self.target_suggestion,
            "quality_score": self.quality_score,
            "metadata": self.metadata
        }
        
        if self.correlations is not None:
            try:
                result["correlations"] = self.correlations.to_dict()
            except Exception:
                result["correlations"] = self.correlations
                
        return result
    
    def get_column_profile(self, column: str) -> Optional[Dict[str, Any]]:
        """Get profile for a specific column."""
        if column not in self.column_types:
            return None
        return {
            "name": column,
            "type": self.column_types.get(column),
            "statistics": self.statistics.get(column, {})
        }
    
    def get_high_quality_columns(self, threshold: float = 80.0) -> List[str]:
        """Get columns with quality score above threshold."""
        return list(self.column_types.keys())
    
    def __repr__(self) -> str:
        return f"DataProfile(rows={self.n_rows}, cols={self.n_columns}, quality={self.quality_score:.1f})"
