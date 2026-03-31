"""Core smoke tests for AutoDS."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        'age': np.random.randint(18, 65, n),
        'income': np.random.normal(50000, 15000, n),
        'score': np.random.uniform(0, 100, n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'target': np.random.choice([0, 1], n),
    })


def test_imports():
    from autods import AutoDS, StatisticalProfiler, auto_preprocess, run_complete_analysis
    assert AutoDS is not None


def test_profiler(sample_df):
    from autods.analysis.profiler import StatisticalProfiler
    profiler = StatisticalProfiler(sample_df)
    profile = profiler.get_profile()
    assert 'overview' in profile
    assert profile['overview']['n_rows'] == 200
    assert 'quality_score' in profile


def test_autods_chain(sample_df):
    from autods import AutoDS
    ads = AutoDS(sample_df, target='target')
    ads.analyze()
    assert ads.profile is not None
    ads.preprocess()
    assert ads.processed_data is not None


def test_preprocessing(sample_df):
    from autods.preprocessing.orchestrator import PreprocessingOrchestrator
    orch = PreprocessingOrchestrator(sample_df, target_column='target')
    result = orch.run_auto(mode='balanced', save=False)
    assert 'data' in result
    assert isinstance(result['data'], pd.DataFrame)


def test_no_duplicate_autopreprocessor():
    """Ensure pipeline.py duplicate is gone."""
    import importlib.util
    spec = importlib.util.find_spec('autods.preprocessing.pipeline')
    assert spec is None, "pipeline.py should be deleted — it was a duplicate"


def test_security_validator_import():
    """Ensure logging import is present in validator."""
    from autods.security.validator import InputValidator
    # This would crash at module import time if logging wasn't imported
    assert InputValidator is not None
