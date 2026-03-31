#!/usr/bin/env python3
"""Final AutoDS Verification Test"""
import sys
sys.path.insert(0, 'src')

print("=" * 70)
print("FINAL AUTODS PROJECT VERIFICATION")
print("=" * 70)

all_passed = True

# Test critical imports from app.py
imports_to_test = [
    ("autods.data.ingestion", "DataIngestion"),
    ("autods.analysis.profiler", "StatisticalProfiler"),
    ("autods.preprocessing.orchestrator", "PreprocessingOrchestrator"),
    ("autods.modeling.orchestrator", "ModelOrchestrator"),
    ("autods.reports.pipeline", "run_complete_analysis"),
    ("autods.utils.error_handler", "ErrorHandler"),
    ("autods.utils.session_manager", "SessionManager"),
    ("autods.utils.checklist", "AnalysisChecklist"),
    ("autods.utils.export_generator", "ExportGenerator"),
    ("autods.utils.demo_data", "DemoDatasetGenerator"),
]

print("\n[TESTING IMPORTS]")
for module, name in imports_to_test:
    try:
        exec(f"from {module} import {name}")
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        all_passed = False

# Test basic functionality
print("\n[TESTING BASIC FUNCTIONALITY]")
try:
    import pandas as pd
    import numpy as np
    
    # Test DataIngestion
    from autods.data.ingestion import DataIngestion
    di = DataIngestion()
    print("  ✓ DataIngestion instantiation")
    
    # Test DemoDatasetGenerator
    from autods.utils.demo_data import DemoDatasetGenerator
    df = DemoDatasetGenerator.generate_customer_churn(n_samples=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    print("  ✓ DemoDatasetGenerator.generate_customer_churn")
    
    # Test StatisticalProfiler
    from autods.analysis.profiler import StatisticalProfiler
    profiler = StatisticalProfiler(df)
    profile = profiler.generate_profile()
    print("  ✓ StatisticalProfiler.generate_profile")
    
except Exception as e:
    print(f"  ✗ Functionality test failed: {e}")
    all_passed = False

print("\n" + "=" * 70)
if all_passed:
    print("✅ ALL TESTS PASSED - PROJECT IS READY!")
    print("\nYou can now run: streamlit run app.py")
    sys.exit(0)
else:
    print("❌ SOME TESTS FAILED - REVIEW ERRORS ABOVE")
    sys.exit(1)
