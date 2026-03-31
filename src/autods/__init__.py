"""AutoDS — Automated Data Science Platform.

Quick start:
    import pandas as pd
    from autods import AutoDS

    df = pd.read_csv('data.csv')
    ads = AutoDS(df, target='churn')
    ads.analyze().preprocess().train()
    print(ads.summary())
    ads.report()

Individual tools:
    from autods import profile, auto_preprocess, full_modeling_pipeline, run_complete_analysis
"""
from autods.core.analyst import AutoDS
from autods.analysis.profiler import StatisticalProfiler
from autods.preprocessing.orchestrator import PreprocessingOrchestrator, auto_preprocess
from autods.modeling.orchestrator import ModelOrchestrator, full_modeling_pipeline
from autods.reports.pipeline import run_complete_analysis
from autods.utils.demo_data import DemoDatasetGenerator

__version__ = "1.0.0"
__author__ = "AutoDS"

__all__ = [
    "AutoDS",
    "StatisticalProfiler",
    "PreprocessingOrchestrator",
    "auto_preprocess",
    "ModelOrchestrator",
    "full_modeling_pipeline",
    "run_complete_analysis",
    "DemoDatasetGenerator",
]