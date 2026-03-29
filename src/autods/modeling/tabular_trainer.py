"""Baseline model auto-training for tabular data."""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
import time
import warnings
warnings.filterwarnings('ignore')

from .task_detector import TaskDetector, MLTaskType
from .recommender import ModelRecommender


class TabularAutoTrainer:
    """Automatically train baseline models for tabular data."""
    
    def __init__(self, 
                 df: pd.DataFrame,
                 target_column: str,
                 test_size: float = 0.2,
                 random_state: int = 42):
        self.df = df.copy()
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        
        # Detect task type
        self.task_detector = TaskDetector(df, target_column)
        self.task_type = self.task_detector.get_task_type()
        self.is_classification = self.task_detector.is_classification()
        self.is_regression = self.task_detector.is_regression()
        
        # Prepare data
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = None
        
        self._prepare_data()
        
        # Results storage
        self.trained_models: Dict[str, Any] = {}
        self.results: List[Dict[str, Any]] = []
    
    def _prepare_data(self):
        """Prepare features and target."""
        # Separate features and target
        self.X = self.df.drop(columns=[self.target_column])
        self.y = self.df[self.target_column]
        
        # Select only numeric features for baseline training
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns
        self.X = self.X[numeric_cols]
        
        # Handle missing values in features
        self.X = self.X.fillna(self.X.median())
        
        # Encode target if classification
        if self.is_classification:
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(self.y)
        
        # Split data
        stratify = self.y if self.is_classification else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=stratify
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def train_baseline_models(self, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train baseline models and return results."""
        if models is None:
            models = self._get_default_models()
        
        print(f"Training baseline models for {self.task_type}...")
        print(f"Data: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print()
        
        for model_name in models:
            try:
                result = self._train_model(model_name)
                if result:
                    self.results.append(result)
                    print(f"✓ {model_name}: {self._format_metric(result)}")
            except Exception as e:
                print(f"✗ {model_name}: Failed - {str(e)[:50]}")
        
        # Sort by main metric
        self.results.sort(key=lambda x: x['main_metric_value'], reverse=True)
        
        return {
            'task_type': self.task_type,
            'data_shape': self.X.shape,
            'models_trained': len(self.results),
            'results': self.results,
            'best_model': self.results[0] if self.results else None
        }
    
    def _get_default_models(self) -> List[str]:
        """Get default models based on task type."""
        if self.is_classification:
            return ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
        elif self.is_regression:
            return ['linear_regression', 'ridge', 'random_forest', 'xgboost', 'lightgbm']
        else:
            return []
    
    def _train_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Train a single model and return results."""
        model = self._create_model(model_name)
        if model is None:
            return None
        
        start_time = time.time()
        
        # Train
        if model_name in ['logistic_regression', 'linear_regression', 'ridge']:
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled) if hasattr(model, 'predict_proba') else None
        else:
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        if self.is_classification:
            metrics = self._calculate_classification_metrics(self.y_test, y_pred, y_pred_proba)
            main_metric = 'f1_weighted'
        else:
            metrics = self._calculate_regression_metrics(self.y_test, y_pred)
            main_metric = 'r2'
        
        # Cross-validation
        cv_scores = self._cross_validate(model, model_name)
        
        return {
            'model_name': model_name,
            'model': model,
            'metrics': metrics,
            'main_metric': main_metric,
            'main_metric_value': metrics[main_metric],
            'training_time': training_time,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores) if cv_scores else None,
            'cv_std': np.std(cv_scores) if cv_scores else None,
        }
    
    def _create_model(self, model_name: str):
        """Create model instance."""
        try:
            if model_name == 'logistic_regression':
                return LogisticRegression(max_iter=1000, random_state=self.random_state)
            elif model_name == 'random_forest':
                if self.is_classification:
                    return RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
                else:
                    return RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            elif model_name == 'xgboost':
                try:
                    import xgboost as xgb
                    if self.is_classification:
                        return xgb.XGBClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
                    else:
                        return xgb.XGBRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
                except ImportError:
                    return None
            elif model_name == 'lightgbm':
                try:
                    import lightgbm as lgb
                    if self.is_classification:
                        return lgb.LGBMClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1, verbose=-1)
                    else:
                        return lgb.LGBMRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1, verbose=-1)
                except ImportError:
                    return None
            elif model_name == 'linear_regression':
                return LinearRegression()
            elif model_name == 'ridge':
                return Ridge(random_state=self.random_state)
        except Exception:
            return None
        return None
    
    def _calculate_classification_metrics(self, y_true, y_pred, y_pred_proba) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # ROC-AUC for binary or multiclass
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            except Exception:
                pass
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100,
        }
    
    def _cross_validate(self, model, model_name: str) -> List[float]:
        """Perform cross-validation."""
        try:
            if self.is_classification:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                scoring = 'f1_weighted'
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
                scoring = 'r2'
            
            if model_name in ['logistic_regression', 'linear_regression', 'ridge']:
                X_cv = self.X_train_scaled
            else:
                X_cv = self.X_train
            
            scores = cross_val_score(model, X_cv, self.y_train, cv=cv, scoring=scoring, n_jobs=-1)
            return scores.tolist()
        except Exception:
            return []
    
    def _format_metric(self, result: Dict) -> str:
        """Format main metric for display."""
        metric = result['main_metric']
        value = result['main_metric_value']
        cv_mean = result.get('cv_mean')
        
        if self.is_classification:
            formatted = f"{metric}={value:.3f}"
        else:
            formatted = f"{metric}={value:.3f}"
        
        if cv_mean:
            formatted += f" (CV: {cv_mean:.3f}±{result.get('cv_std', 0):.3f})"
        
        return formatted
    
    def get_best_model(self) -> Optional[Any]:
        """Get the best trained model."""
        if self.results:
            return self.results[0]['model']
        return None
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with best model."""
        best_model_info = self.get_best_model()
        if best_model_info is None:
            raise ValueError("No models trained yet")
        
        model = best_model_info if not isinstance(best_model_info, dict) else best_model_info.get('model')
        
        # Select numeric features only
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_clean = X[numeric_cols].fillna(X[numeric_cols].median())
        
        predictions = model.predict(X_clean)
        
        # Decode if classification
        if self.is_classification and self.label_encoder:
            predictions = self.label_encoder.inverse_transform(predictions.astype(int))
        
        return predictions
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of training results."""
        if not self.results:
            return {'error': 'No models trained'}
        
        best = self.results[0]
        
        return {
            'task_type': self.task_type,
            'total_models': len(self.results),
            'best_model_name': best['model_name'],
            'best_score': best['main_metric_value'],
            'all_models': [
                {
                    'name': r['model_name'],
                    'score': r['main_metric_value'],
                    'cv_mean': r.get('cv_mean'),
                    'training_time': r['training_time']
                }
                for r in self.results
            ]
        }


class QuickTrainer:
    """Quick training without full configuration."""
    
    @staticmethod
    def train(df: pd.DataFrame, target: str, model_type: str = 'auto') -> Dict[str, Any]:
        """Quick one-line training."""
        trainer = TabularAutoTrainer(df, target)
        
        if model_type == 'auto':
            return trainer.train_baseline_models()
        else:
            return trainer.train_baseline_models([model_type])
    
    @staticmethod
    def compare_models(df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Quickly compare all baseline models."""
        results = QuickTrainer.train(df, target)
        
        comparison = []
        for r in results['results']:
            comparison.append({
                'Model': r['model_name'],
                'Score': round(r['main_metric_value'], 4),
                'CV Mean': round(r.get('cv_mean', 0), 4) if r.get('cv_mean') else 'N/A',
                'CV Std': round(r.get('cv_std', 0), 4) if r.get('cv_std') else 'N/A',
                'Time (s)': round(r['training_time'], 2)
            })
        
        return pd.DataFrame(comparison)
