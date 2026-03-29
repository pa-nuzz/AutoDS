"""Model recommendation engine with reasoning."""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from .task_detector import TaskDetector, MLTaskType


class ModelRecommender:
    """Recommend appropriate ML models with reasoning."""
    
    # Model definitions with metadata
    MODELS = {
        # Tabular - Classification
        'random_forest_classifier': {
            'name': 'Random Forest Classifier',
            'type': 'tabular',
            'task_types': [MLTaskType.BINARY_CLASSIFICATION, MLTaskType.MULTICLASS_CLASSIFICATION],
            'complexity': 'medium',
            'interpretability': 'high',
            'handles_missing': True,
            'handles_categorical': False,  # Needs encoding
            'handles_outliers': True,
            'training_speed': 'fast',
            'inference_speed': 'fast',
            'memory_usage': 'medium',
            'best_for': ['mixed_feature_types', 'feature_importance_needed', 'robust_to_outliers'],
            'pros': ['Handles mixed data types', 'Robust to outliers', 'Built-in feature importance', 'No overfitting with proper tuning'],
            'cons': ['Can overfit with deep trees', 'Slow with many trees', 'High memory for large datasets'],
        },
        'xgboost_classifier': {
            'name': 'XGBoost Classifier',
            'type': 'tabular',
            'task_types': [MLTaskType.BINARY_CLASSIFICATION, MLTaskType.MULTICLASS_CLASSIFICATION],
            'complexity': 'medium',
            'interpretability': 'medium',
            'handles_missing': True,
            'handles_categorical': False,
            'handles_outliers': True,
            'training_speed': 'medium',
            'inference_speed': 'fast',
            'memory_usage': 'medium',
            'best_for': ['high_performance', 'heterogeneous_data', 'large_datasets'],
            'pros': ['Excellent accuracy', 'Handles missing values', 'Built-in regularization', 'Fast prediction'],
            'cons': ['Requires hyperparameter tuning', 'Can overfit', 'Less interpretable than RF'],
        },
        'lightgbm_classifier': {
            'name': 'LightGBM Classifier',
            'type': 'tabular',
            'task_types': [MLTaskType.BINARY_CLASSIFICATION, MLTaskType.MULTICLASS_CLASSIFICATION],
            'complexity': 'medium',
            'interpretability': 'medium',
            'handles_missing': True,
            'handles_categorical': True,
            'handles_outliers': True,
            'training_speed': 'fast',
            'inference_speed': 'fast',
            'memory_usage': 'low',
            'best_for': ['large_datasets', 'high_cardinality_features', 'speed_critical'],
            'pros': ['Very fast training', 'Low memory usage', 'Handles categorical natively', 'Great for large data'],
            'cons': ['Sensitive to overfitting', 'Requires careful tuning', 'Leaf-wise growth can overfit small data'],
        },
        'logistic_regression': {
            'name': 'Logistic Regression',
            'type': 'tabular',
            'task_types': [MLTaskType.BINARY_CLASSIFICATION, MLTaskType.MULTICLASS_CLASSIFICATION],
            'complexity': 'low',
            'interpretability': 'high',
            'handles_missing': False,
            'handles_categorical': False,
            'handles_outliers': False,
            'training_speed': 'very_fast',
            'inference_speed': 'very_fast',
            'memory_usage': 'very_low',
            'best_for': ['baseline', 'interpretability', 'linear_separable', 'small_datasets'],
            'pros': ['Highly interpretable', 'Fast training and inference', 'Well-calibrated probabilities', 'No hyperparameter tuning'],
            'cons': ['Assumes linear relationship', 'Sensitive to outliers', 'Requires feature scaling', 'No feature interactions'],
        },
        # Tabular - Regression
        'random_forest_regressor': {
            'name': 'Random Forest Regressor',
            'type': 'tabular',
            'task_types': [MLTaskType.REGRESSION],
            'complexity': 'medium',
            'interpretability': 'high',
            'handles_missing': True,
            'handles_categorical': False,
            'handles_outliers': True,
            'training_speed': 'fast',
            'inference_speed': 'fast',
            'memory_usage': 'medium',
            'best_for': ['mixed_feature_types', 'feature_importance_needed', 'robust_to_outliers'],
            'pros': ['Handles non-linearity', 'Robust to outliers', 'Built-in feature importance', 'No overfitting with proper tuning'],
            'cons': ['Slow with many trees', 'High memory usage', 'Poor extrapolation'],
        },
        'xgboost_regressor': {
            'name': 'XGBoost Regressor',
            'type': 'tabular',
            'task_types': [MLTaskType.REGRESSION],
            'complexity': 'medium',
            'interpretability': 'medium',
            'handles_missing': True,
            'handles_categorical': False,
            'handles_outliers': True,
            'training_speed': 'medium',
            'inference_speed': 'fast',
            'memory_usage': 'medium',
            'best_for': ['high_performance', 'heterogeneous_data', 'large_datasets'],
            'pros': ['Excellent accuracy', 'Handles missing values', 'Built-in regularization', 'Handles non-linearity'],
            'cons': ['Requires hyperparameter tuning', 'Can overfit', 'Less interpretable'],
        },
        'lightgbm_regressor': {
            'name': 'LightGBM Regressor',
            'type': 'tabular',
            'task_types': [MLTaskType.REGRESSION],
            'complexity': 'medium',
            'interpretability': 'medium',
            'handles_missing': True,
            'handles_categorical': True,
            'handles_outliers': True,
            'training_speed': 'fast',
            'inference_speed': 'fast',
            'memory_usage': 'low',
            'best_for': ['large_datasets', 'high_cardinality_features', 'speed_critical'],
            'pros': ['Very fast training', 'Low memory', 'Handles categorical natively', 'Great performance'],
            'cons': ['Sensitive to overfitting', 'Requires tuning'],
        },
        'linear_regression': {
            'name': 'Linear Regression',
            'type': 'tabular',
            'task_types': [MLTaskType.REGRESSION],
            'complexity': 'low',
            'interpretability': 'very_high',
            'handles_missing': False,
            'handles_categorical': False,
            'handles_outliers': False,
            'training_speed': 'very_fast',
            'inference_speed': 'very_fast',
            'memory_usage': 'very_low',
            'best_for': ['baseline', 'interpretability', 'linear_relationship', 'small_datasets'],
            'pros': ['Highly interpretable', 'Fast', 'No tuning needed', 'Well-understood'],
            'cons': ['Assumes linearity', 'Sensitive to outliers', 'Requires scaling', 'No interactions'],
        },
        'ridge_regression': {
            'name': 'Ridge Regression',
            'type': 'tabular',
            'task_types': [MLTaskType.REGRESSION],
            'complexity': 'low',
            'interpretability': 'high',
            'handles_missing': False,
            'handles_categorical': False,
            'handles_outliers': False,
            'training_speed': 'fast',
            'inference_speed': 'very_fast',
            'memory_usage': 'low',
            'best_for': ['multicollinearity', 'regularization_needed', 'interpretability'],
            'pros': ['Handles multicollinearity', 'L2 regularization prevents overfitting', 'Fast', 'Interpretable'],
            'cons': ['Assumes linearity', 'Sensitive to outliers'],
        },
        # Clustering
        'kmeans': {
            'name': 'K-Means Clustering',
            'type': 'clustering',
            'task_types': [MLTaskType.CLUSTERING],
            'complexity': 'low',
            'interpretability': 'high',
            'handles_missing': False,
            'handles_categorical': False,
            'handles_outliers': False,
            'training_speed': 'fast',
            'inference_speed': 'very_fast',
            'memory_usage': 'low',
            'best_for': ['spherical_clusters', 'large_datasets', 'baseline_clustering'],
            'pros': ['Simple and fast', 'Scales well', 'Easy to understand'],
            'cons': ['Requires specifying K', 'Sensitive to outliers', 'Assumes spherical clusters', 'Needs scaling'],
        },
        'dbscan': {
            'name': 'DBSCAN',
            'type': 'clustering',
            'task_types': [MLTaskType.CLUSTERING],
            'complexity': 'medium',
            'interpretability': 'medium',
            'handles_missing': False,
            'handles_categorical': False,
            'handles_outliers': True,
            'training_speed': 'medium',
            'inference_speed': 'fast',
            'memory_usage': 'medium',
            'best_for': ['irregular_clusters', 'noise_detection', 'unknown_n_clusters'],
            'pros': ['Finds arbitrary shapes', 'Handles noise/outliers', 'No need to specify K'],
            'cons': ['Sensitive to parameters', 'Struggles with varying density', 'Needs scaling'],
        },
        # NLP
        'tfidf_logistic': {
            'name': 'TF-IDF + Logistic Regression',
            'type': 'nlp',
            'task_types': [MLTaskType.NLP_CLASSIFICATION, MLTaskType.NLP_REGRESSION],
            'complexity': 'low',
            'interpretability': 'high',
            'handles_missing': True,
            'handles_categorical': True,
            'handles_outliers': True,
            'training_speed': 'fast',
            'inference_speed': 'fast',
            'memory_usage': 'medium',
            'best_for': ['baseline_nlp', 'interpretability', 'small_medium_text'],
            'pros': ['Fast and simple', 'Interpretable coefficients', 'Good baseline', 'Works with short texts'],
            'cons': ['No semantic understanding', 'Bag-of-words limitations', 'Sparsity issues with large vocab'],
        },
        'bert_classifier': {
            'name': 'BERT (Transformer)',
            'type': 'nlp',
            'task_types': [MLTaskType.NLP_CLASSIFICATION, MLTaskType.NLP_REGRESSION],
            'complexity': 'high',
            'interpretability': 'low',
            'handles_missing': True,
            'handles_categorical': True,
            'handles_outliers': True,
            'training_speed': 'slow',
            'inference_speed': 'medium',
            'memory_usage': 'high',
            'best_for': ['high_accuracy', 'semantic_understanding', 'sufficient_data', 'gpu_available'],
            'pros': ['State-of-the-art accuracy', 'Deep semantic understanding', 'Transfer learning', 'Context aware'],
            'cons': ['Requires GPU for training', 'Slow inference', 'High memory', 'Needs more data', 'Black box'],
        },
        # Image
        'cnn_baseline': {
            'name': 'Simple CNN',
            'type': 'image',
            'task_types': [MLTaskType.IMAGE_CLASSIFICATION, MLTaskType.IMAGE_REGRESSION],
            'complexity': 'medium',
            'interpretability': 'low',
            'handles_missing': True,
            'handles_categorical': True,
            'handles_outliers': True,
            'training_speed': 'medium',
            'inference_speed': 'fast',
            'memory_usage': 'medium',
            'best_for': ['baseline_image', 'small_dataset', 'custom_architecture'],
            'pros': ['Fast training', 'Simple architecture', 'Good for learning', 'Customizable'],
            'cons': ['Lower accuracy than pretrained', 'Requires more data', 'Overfitting risk'],
        },
        'resnet': {
            'name': 'ResNet (Pretrained)',
            'type': 'image',
            'task_types': [MLTaskType.IMAGE_CLASSIFICATION, MLTaskType.IMAGE_REGRESSION],
            'complexity': 'high',
            'interpretability': 'very_low',
            'handles_missing': True,
            'handles_categorical': True,
            'handles_outliers': True,
            'training_speed': 'fast',  # Fast with transfer learning
            'inference_speed': 'fast',
            'memory_usage': 'high',
            'best_for': ['high_accuracy', 'transfer_learning', 'limited_data', 'gpu_available'],
            'pros': ['Excellent accuracy', 'Transfer learning', 'Works with limited data', 'Proven architecture'],
            'cons': ['Requires GPU', 'High memory', 'Black box', 'ImageNet bias'],
        },
        'efficientnet': {
            'name': 'EfficientNet',
            'type': 'image',
            'task_types': [MLTaskType.IMAGE_CLASSIFICATION, MLTaskType.IMAGE_REGRESSION],
            'complexity': 'high',
            'interpretability': 'very_low',
            'handles_missing': True,
            'handles_categorical': True,
            'handles_outliers': True,
            'training_speed': 'medium',
            'inference_speed': 'fast',
            'memory_usage': 'medium',
            'best_for': ['efficiency', 'mobile_deployment', 'accuracy_efficiency_tradeoff'],
            'pros': ['Best accuracy/efficiency tradeoff', 'Mobile-friendly', 'Transfer learning', 'Scalable'],
            'cons': ['Complex architecture', 'Requires tuning', 'Black box'],
        },
    }
    
    def __init__(self, task_detector: TaskDetector, dataset_info: Optional[Dict] = None):
        self.task_detector = task_detector
        self.task_type = task_detector.get_task_type()
        self.dataset_info = dataset_info or {}
        self.recommendations: List[Dict[str, Any]] = []
        self._generate_recommendations()
    
    def _generate_recommendations(self):
        """Generate model recommendations based on task type and dataset characteristics."""
        # Filter models by task type
        suitable_models = [
            (key, model) for key, model in self.MODELS.items()
            if self.task_type in model['task_types']
        ]
        
        # Score each model
        scored_models = []
        for key, model in suitable_models:
            score = self._score_model(model)
            scored_models.append((key, model, score))
        
        # Sort by score descending
        scored_models.sort(key=lambda x: x[2], reverse=True)
        
        # Generate recommendations with reasoning
        for rank, (key, model, score) in enumerate(scored_models[:5], 1):
            recommendation = {
                'rank': rank,
                'model_key': key,
                'model_name': model['name'],
                'confidence_score': score,
                'model_info': model,
                'reasoning': self._generate_reasoning(model, score),
                'estimated_performance': self._estimate_performance(model),
            }
            self.recommendations.append(recommendation)
    
    def _score_model(self, model: Dict) -> float:
        """Score a model based on dataset characteristics."""
        score = 100.0  # Base score
        
        # Dataset size factor
        n_samples = self.dataset_info.get('n_samples', 1000)
        if n_samples < 1000:
            # Small dataset - prefer simpler models
            if model['complexity'] == 'low':
                score += 20
            elif model['complexity'] == 'high':
                score -= 30
        elif n_samples > 100000:
            # Large dataset - prefer fast, scalable models
            if model['training_speed'] == 'fast':
                score += 15
            if model['memory_usage'] in ['low', 'very_low']:
                score += 10
        
        # Missing values factor
        if self.dataset_info.get('has_missing_values'):
            if model['handles_missing']:
                score += 15
            else:
                score -= 20
        
        # Outliers factor
        if self.dataset_info.get('has_outliers'):
            if model['handles_outliers']:
                score += 10
            else:
                score -= 15
        
        # Interpretability preference
        if self.dataset_info.get('needs_interpretability'):
            interpretability_scores = {
                'very_high': 20,
                'high': 15,
                'medium': 5,
                'low': -10,
                'very_low': -20
            }
            score += interpretability_scores.get(model['interpretability'], 0)
        
        # Speed requirements
        if self.dataset_info.get('speed_critical'):
            if model['training_speed'] in ['very_fast', 'fast']:
                score += 15
            if model['inference_speed'] in ['very_fast', 'fast']:
                score += 10
        
        return max(0, score)
    
    def _generate_reasoning(self, model: Dict, score: float) -> str:
        """Generate human-readable reasoning for the recommendation."""
        reasons = []
        
        # Primary reason based on best_for attributes
        if model['best_for']:
            best_match = model['best_for'][0].replace('_', ' ').title()
            reasons.append(f"Best for: {best_match}")
        
        # Data handling capabilities
        handling = []
        if model['handles_missing']:
            handling.append("handles missing values")
        if model['handles_outliers']:
            handling.append("robust to outliers")
        if handling:
            reasons.append(f"Data handling: {', '.join(handling)}")
        
        # Speed characteristics
        speed_info = f"Training: {model['training_speed']}, Inference: {model['inference_speed']}"
        reasons.append(speed_info)
        
        # Complexity vs interpretability
        reasons.append(f"Complexity: {model['complexity']}, Interpretability: {model['interpretability']}")
        
        # Key pros
        if model['pros']:
            reasons.append(f"Key advantages: {', '.join(model['pros'][:2])}")
        
        return " | ".join(reasons)
    
    def _estimate_performance(self, model: Dict) -> Dict[str, str]:
        """Estimate expected performance level."""
        # This is a heuristic based on model type and typical benchmarks
        if model['name'] in ['XGBoost Classifier', 'XGBoost Regressor', 
                             'LightGBM Classifier', 'LightGBM Regressor',
                             'BERT (Transformer)', 'ResNet (Pretrained)', 'EfficientNet']:
            return {'expected': 'high', 'confidence': 'medium'}
        elif model['name'] in ['Random Forest Classifier', 'Random Forest Regressor']:
            return {'expected': 'medium-high', 'confidence': 'high'}
        elif model['complexity'] == 'low':
            return {'expected': 'baseline', 'confidence': 'high'}
        else:
            return {'expected': 'medium', 'confidence': 'medium'}
    
    def get_recommendations(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """Get top k model recommendations."""
        return self.recommendations[:top_k]
    
    def get_best_model(self) -> Optional[Dict[str, Any]]:
        """Get the top recommended model."""
        return self.recommendations[0] if self.recommendations else None
    
    def get_recommendation_summary(self) -> Dict[str, Any]:
        """Get summary of all recommendations."""
        return {
            'task_type': self.task_type,
            'task_description': self.task_detector.task_info.get('description'),
            'n_recommendations': len(self.recommendations),
            'top_models': [
                {
                    'rank': r['rank'],
                    'name': r['model_name'],
                    'score': round(r['confidence_score'], 1),
                    'reasoning': r['reasoning']
                }
                for r in self.recommendations[:3]
            ]
        }
