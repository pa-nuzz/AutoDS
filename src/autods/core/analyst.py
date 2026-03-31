"""Core AutoDS analyst — main library entry point."""
from typing import Dict, Any, Optional
import pandas as pd

from ..analysis.profiler import StatisticalProfiler
from ..analysis.insights import InsightEngine
from ..preprocessing.orchestrator import PreprocessingOrchestrator
from ..modeling.orchestrator import ModelOrchestrator


class AutoDS:
    """Main AutoDS class for automated data science workflows.
    
    Usage:
        import pandas as pd
        from autods import AutoDS
        
        df = pd.read_csv('data.csv')
        ads = AutoDS(df, target='churn')
        ads.analyze()      # Profile + insights
        ads.preprocess()   # Clean and prepare
        ads.train()        # Train baseline models
        ads.report()       # Generate report
    """
    
    def __init__(self, df: pd.DataFrame, target: Optional[str] = None):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df)}")
        if len(df) == 0:
            raise ValueError("DataFrame is empty")
        
        self.df = df.copy()
        self.target = target
        self._profile: Optional[Dict] = None
        self._insights = None
        self._processed_df: Optional[pd.DataFrame] = None
        self._model_results: Optional[Dict] = None
    
    def analyze(self) -> 'AutoDS':
        """Profile and generate insights from the data."""
        profiler = StatisticalProfiler(self.df)
        self._profile = profiler.get_profile()
        self._insights = InsightEngine(profiler)
        return self  # chainable
    
    def preprocess(self, mode: str = 'balanced') -> 'AutoDS':
        """Preprocess the data."""
        if self._profile is None:
            self.analyze()
        orch = PreprocessingOrchestrator(self.df, self.target)
        result = orch.run_auto(mode=mode, save=False)
        self._processed_df = result['data']
        return self
    
    def train(self, models: Optional[list] = None) -> 'AutoDS':
        """Train baseline models."""
        if self.target is None:
            raise ValueError("target must be set for training. Use AutoDS(df, target='col')")
        df_to_use = self._processed_df if self._processed_df is not None else self.df
        orch = ModelOrchestrator(df_to_use, self.target)
        self._model_results = orch.train_baseline(models=models)
        return self
    
    def report(self, output_dir: str = 'autods_output') -> Dict[str, Any]:
        """Generate and export full analysis report."""
        from ..reports.pipeline import run_complete_analysis
        df_to_use = self._processed_df if self._processed_df is not None else self.df
        return run_complete_analysis(df_to_use, target=self.target, output_dir=output_dir)
    
    @property
    def profile(self) -> Optional[Dict]:
        return self._profile
    
    @property
    def insights(self):
        return self._insights
    
    @property
    def processed_data(self) -> Optional[pd.DataFrame]:
        return self._processed_df
    
    @property
    def model_results(self) -> Optional[Dict]:
        return self._model_results
    
    def summary(self) -> str:
        """Print a text summary of current state."""
        lines = [f"AutoDS — {len(self.df):,} rows × {len(self.df.columns)} columns"]
        if self.target:
            lines.append(f"Target: {self.target}")
        if self._profile:
            q = self._profile.get('quality_score', {})
            lines.append(f"Quality score: {q.get('overall', 'N/A')}/100")
        if self._processed_df is not None:
            lines.append(f"Processed shape: {self._processed_df.shape}")
        if self._model_results and self._model_results.get('best_model'):
            bm = self._model_results['best_model']
            lines.append(f"Best model: {bm['model_name']} ({bm['main_metric']}: {bm['main_metric_value']:.4f})")
        return '\n'.join(lines)
    
    def __repr__(self):
        return f"AutoDS(rows={len(self.df)}, cols={len(self.df.columns)}, target={self.target!r})"
