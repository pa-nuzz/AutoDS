"""Main preprocessing orchestrator - combines auto and DIY modes."""
from typing import Dict, Any, List, Optional, Union, Literal
import pandas as pd
from pathlib import Path

from .detector import PreprocessingDetector, PreprocessingNeed, NeedSeverity
from .auto_processor import AutoPreprocessor
from .diy_guide import DIYGuide


class PreprocessingOrchestrator:
    """Main interface for preprocessing - supports both auto and DIY modes."""
    
    MODE_AUTO = "auto"  # Do it for me
    MODE_DIY = "diy"    # Show me how to do it
    
    def __init__(self, 
                 df: pd.DataFrame,
                 target_column: Optional[str] = None,
                 output_dir: str = "processed_data"):
        self.df_original = df.copy()
        self.target_column = target_column
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize detector
        self.detector = PreprocessingDetector(df, target_column)
        
        # Results storage
        self.auto_result = None
        self.diy_guide = None
    
    def run_auto(self, 
                 mode: Literal["minimal", "balanced", "aggressive"] = "balanced",
                 save: bool = True) -> Dict[str, Any]:
        """Run 'Do it for me' auto-preprocessing mode.
        
        Args:
            mode: 'minimal' (essential only), 'balanced' (recommended), 'aggressive' (all)
            save: Whether to save processed data and preprocessor state
            
        Returns:
            Dictionary with processed data, summary, and file paths
        """
        print(f"Running auto-preprocessing (mode: {mode})...")
        
        # Check if auto-processing is safe
        summary = self.detector.get_summary()
        if not summary['can_auto_process']:
            print("⚠️  Warning: Dataset has complex needs that may require manual review.")
            print("Consider using DIY mode for better control.")
        
        # Run auto preprocessor
        preprocessor = AutoPreprocessor(
            self.df_original,
            target_column=self.target_column,
            mode=mode
        )
        df_processed = preprocessor.fit_transform()
        
        result = {
            'mode': 'auto',
            'preprocessing_mode': mode,
            'data': df_processed,
            'summary': preprocessor.get_summary(),
            'steps_applied': preprocessor.pipeline_steps,
            'needs_detected': summary['total_needs'],
        }
        
        # Save if requested
        if save:
            data_path = self.output_dir / "processed_data.csv"
            df_processed.to_csv(data_path, index=False)
            result['data_path'] = str(data_path)
            
            preprocessor.save(self.output_dir / "preprocessor_state.pkl")
            result['preprocessor_path'] = str(self.output_dir / "preprocessor_state.pkl")
            
            # Save summary
            import json
            with open(self.output_dir / "preprocessing_summary.json", 'w') as f:
                json.dump(preprocessor.get_summary(), f, indent=2, default=str)
            
            print(f"✓ Saved processed data to: {data_path}")
        
        self.auto_result = result
        return result
    
    def run_diy(self, save: bool = True) -> Dict[str, Any]:
        """Run 'I want to do it myself' DIY mode.
        
        Returns:
            Dictionary with step-by-step guide and visualizations
        """
        print("Generating DIY preprocessing guide...")
        
        guide = DIYGuide(self.detector)
        
        result = {
            'mode': 'diy',
            'needs_detected': self.detector.get_summary(),
            'steps': guide.get_steps(),
            'markdown_guide': guide.get_markdown_guide(),
        }
        
        # Save guide if requested
        if save:
            guide_path = self.output_dir / "preprocessing_guide.md"
            guide.save_guide(str(guide_path))
            result['guide_path'] = str(guide_path)
            print(f"✓ Saved guide to: {guide_path}")
        
        self.diy_guide = result
        return result
    
    def get_needs_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing needs."""
        return {
            'summary': self.detector.get_summary(),
            'all_needs': self.detector.get_needs(),
            'by_type': {
                need_type: self.detector.get_needs_by_type(need_type)
                for need_type in set(n['type'] for n in self.detector.get_needs())
            }
        }
    
    def compare_modes(self) -> Dict[str, Any]:
        """Compare auto vs DIY recommendations."""
        summary = self.detector.get_summary()
        
        return {
            'auto_suitable': summary['can_auto_process'],
            'recommendation': 'auto' if summary['can_auto_process'] and summary['required'] <= 3 else 'diy',
            'reasoning': (
                "Auto-preprocessing is recommended" if summary['can_auto_process'] 
                else "DIY mode recommended due to complex preprocessing needs"
            ),
            'needs_summary': summary,
            'auto_estimate': {
                'steps': summary['total_needs'],
                'time': f"~{summary['total_needs'] * 2} minutes",
            }
        }


class QuickPreprocessor:
    """Quick preprocessing for common scenarios."""
    
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """Quick clean - remove duplicates and handle missing values."""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Drop columns with >50% missing
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > 0.5].index
        df = df.drop(columns=cols_to_drop)
        
        # Simple imputation for remaining
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        return df
    
    @staticmethod
    def prepare_numeric(df: pd.DataFrame, target: Optional[str] = None) -> pd.DataFrame:
        """Quick numeric preparation - encode categoricals and scale."""
        # Select only numeric and categorical
        df = df.select_dtypes(include=['number', 'object', 'category'])
        
        # One-hot encode categoricals with low cardinality
        cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns 
                   if df[c].nunique() <= 10 and c != target]
        
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        
        # Drop remaining high-cardinality categoricals
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        df = df.drop(columns=cat_cols)
        
        # Scale numeric
        numeric_cols = df.select_dtypes(include=['number']).columns
        if target in numeric_cols:
            numeric_cols = numeric_cols.drop(target)
        
        if len(numeric_cols) > 0:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        return df
    
    @staticmethod
    def full_pipeline(df: pd.DataFrame, 
                     target: Optional[str] = None,
                     mode: str = "balanced") -> pd.DataFrame:
        """Complete preprocessing pipeline."""
        orchestrator = PreprocessingOrchestrator(df, target)
        result = orchestrator.run_auto(mode=mode, save=False)
        return result['data']


# Convenience functions
def auto_preprocess(df: pd.DataFrame, 
                   target: Optional[str] = None,
                   mode: str = "balanced") -> pd.DataFrame:
    """One-line auto preprocessing."""
    orchestrator = PreprocessingOrchestrator(df, target)
    result = orchestrator.run_auto(mode=mode, save=False)
    return result['data']


def get_preprocessing_guide(df: pd.DataFrame, 
                           target: Optional[str] = None) -> str:
    """Get DIY preprocessing guide as markdown string."""
    orchestrator = PreprocessingOrchestrator(df, target)
    result = orchestrator.run_diy(save=False)
    return result['markdown_guide']


def analyze_preprocessing_needs(df: pd.DataFrame, 
                              target: Optional[str] = None) -> Dict[str, Any]:
    """Analyze what preprocessing is needed."""
    orchestrator = PreprocessingOrchestrator(df, target)
    return orchestrator.get_needs_summary()
