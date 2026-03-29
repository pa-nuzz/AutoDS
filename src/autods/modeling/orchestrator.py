"""Main modeling orchestrator combining task detection, recommendations, and training."""
from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path

from .task_detector import TaskDetector, MLTaskType
from .recommender import ModelRecommender
from .tabular_trainer import TabularAutoTrainer, QuickTrainer
from ..analysis.ai_enhancement import AIDataSummarizer, LLMClient


class ModelOrchestrator:
    """Main interface for model recommendation and training."""
    
    def __init__(self,
                 df: pd.DataFrame,
                 target_column: Optional[str] = None,
                 output_dir: str = "reports/models",
                 use_llm: bool = False):
        self.df = df
        self.target_column = target_column
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Task detection
        self.task_detector = TaskDetector(df, target_column)
        self.task_type = self.task_detector.get_task_type()
        
        # Dataset info for recommender
        self.dataset_info = self._gather_dataset_info()
        
        # Model recommendation
        self.recommender = ModelRecommender(self.task_detector, self.dataset_info)
        
        # Results storage
        self.training_results = None
        self.ai_explanation = None
        
        # AI enhancement
        self.ai_summarizer = None
        if use_llm:
            try:
                llm_client = LLMClient()
                self.ai_summarizer = AIDataSummarizer(llm_client, use_llm=True)
            except Exception:
                pass
    
    def _gather_dataset_info(self) -> Dict[str, Any]:
        """Gather dataset information for model scoring."""
        n_samples = len(self.df)
        n_features = len(self.df.columns) - (1 if self.target_column else 0)
        
        # Check for missing values
        missing_pct = self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100
        has_missing = missing_pct > 0
        
        # Check numeric columns for outliers
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        has_outliers = False
        for col in numeric_cols:
            if col == self.target_column:
                continue
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)).sum()
            if outliers / len(self.df) > 0.05:  # >5% outliers
                has_outliers = True
                break
        
        return {
            'n_samples': n_samples,
            'n_features': n_features,
            'has_missing_values': has_missing,
            'missing_percentage': missing_pct,
            'has_outliers': has_outliers,
            'needs_interpretability': n_features < 50,  # Assume interpretability needed for smaller datasets
            'speed_critical': n_samples > 100000,
        }
    
    def get_recommendations(self, top_k: int = 3) -> Dict[str, Any]:
        """Get model recommendations."""
        recommendations = self.recommender.get_recommendations(top_k)
        summary = self.recommender.get_recommendation_summary()
        
        # Generate AI explanation if available
        ai_explanation = None
        if self.ai_summarizer:
            ai_explanation = self._generate_ai_recommendation_explanation(recommendations)
        
        return {
            'task_type': self.task_type,
            'task_info': self.task_detector.get_task_info(),
            'dataset_info': self.dataset_info,
            'recommendations': recommendations,
            'summary': summary,
            'ai_explanation': ai_explanation,
        }
    
    def train_baseline(self, models: Optional[List[str]] = None, save: bool = True) -> Dict[str, Any]:
        """Train baseline models."""
        if not self.target_column:
            return {'error': 'No target column specified for supervised learning'}
        
        # Check if tabular
        if self.task_type in [MLTaskType.REGRESSION, MLTaskType.BINARY_CLASSIFICATION, 
                              MLTaskType.MULTICLASS_CLASSIFICATION]:
            trainer = TabularAutoTrainer(self.df, self.target_column)
            results = trainer.train_baseline_models(models)
            self.training_results = results
            
            if save and results.get('results'):
                self._save_training_results(results)
            
            return results
        else:
            return {
                'error': f'Auto-training not yet implemented for {self.task_type}',
                'task_type': self.task_type,
                'message': 'Model recommendations available, but auto-training requires manual implementation'
            }
    
    def _generate_ai_recommendation_explanation(self, recommendations: List[Dict]) -> Optional[str]:
        """Generate AI explanation for recommendations."""
        if not self.ai_summarizer:
            return None
        
        try:
            task_info = self.task_detector.get_task_info()
            
            prompt = f"""As a machine learning expert, explain why these model recommendations are appropriate for this task:

TASK: {task_info.get('description', self.task_type)}
DATASET: {self.dataset_info['n_samples']} samples, {self.dataset_info['n_features']} features
MISSING VALUES: {self.dataset_info['missing_percentage']:.1f}%
OUTLIERS DETECTED: {self.dataset_info['has_outliers']}

TOP RECOMMENDED MODELS:
"""
            for i, rec in enumerate(recommendations[:3], 1):
                prompt += f"""
{i}. {rec['model_name']}
   - Reasoning: {rec['reasoning']}
   - Expected Performance: {rec['estimated_performance']['expected']}
"""
            
            prompt += """
Please provide:
1. A brief explanation of why the top model is recommended
2. Key factors that influenced the recommendation
3. When to consider the alternative models
4. Expected challenges with this dataset type

Be concise and actionable."""
            
            return self.ai_summarizer.llm.generate(prompt)
        except Exception:
            return None
    
    def _save_training_results(self, results: Dict[str, Any]):
        """Save training results to files."""
        import json
        
        # Save JSON results
        summary = {
            'task_type': results['task_type'],
            'data_shape': results['data_shape'],
            'models_trained': results['models_trained'],
            'best_model': {
                'name': results['best_model']['model_name'],
                'main_metric': results['best_model']['main_metric'],
                'score': results['best_model']['main_metric_value'],
                'cv_mean': results['best_model'].get('cv_mean'),
            } if results['best_model'] else None,
            'all_results': [
                {
                    'model': r['model_name'],
                    'score': r['main_metric_value'],
                    'metrics': r['metrics'],
                    'cv_mean': r.get('cv_mean'),
                    'training_time': r['training_time']
                }
                for r in results['results']
            ]
        }
        
        with open(self.output_dir / 'training_results.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def get_full_report(self) -> Dict[str, Any]:
        """Get comprehensive modeling report."""
        recommendations = self.get_recommendations()
        
        report = {
            'task_detection': {
                'type': self.task_type,
                'info': self.task_detector.get_task_info(),
                'is_supervised': self.task_detector.is_supervised(),
                'is_classification': self.task_detector.is_classification(),
                'is_regression': self.task_detector.is_regression(),
            },
            'model_recommendations': recommendations,
            'dataset_characteristics': self.dataset_info,
        }
        
        if self.training_results:
            report['training_results'] = self.training_results
        
        return report
    
    def save_report(self, filepath: Optional[str] = None) -> str:
        """Save full report to file."""
        import json
        
        if filepath is None:
            filepath = self.output_dir / 'model_report.json'
        
        report = self.get_full_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(filepath)
    
    def compare_modes(self) -> Dict[str, str]:
        """Compare quick vs detailed approach."""
        return {
            'quick': 'Use QuickTrainer.train(df, target) for fast baseline',
            'full': 'Use ModelOrchestrator for recommendations + training + explanations',
            'recommendation': 'full' if self.dataset_info['n_samples'] > 1000 else 'quick'
        }


# Convenience functions
def recommend_models(df: pd.DataFrame, 
                    target: Optional[str] = None,
                    use_llm: bool = False) -> Dict[str, Any]:
    """Quick model recommendations."""
    orchestrator = ModelOrchestrator(df, target, use_llm=use_llm)
    return orchestrator.get_recommendations()


def train_baselines(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """Quick baseline training."""
    return QuickTrainer.train(df, target)


def full_modeling_pipeline(df: pd.DataFrame, 
                          target: str,
                          use_llm: bool = False) -> Dict[str, Any]:
    """Complete modeling pipeline: detect, recommend, train."""
    orchestrator = ModelOrchestrator(df, target, use_llm=use_llm)
    
    results = {
        'recommendations': orchestrator.get_recommendations(),
        'training': orchestrator.train_baseline(),
    }
    
    return results
