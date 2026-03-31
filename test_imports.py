#!/usr/bin/env python3
"""Comprehensive AutoDS Import Test"""
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("AutoDS Comprehensive Import Test")
print("=" * 60)

errors = []
warnings = []

# Test 1: Core Data Imports
print("\n[1] Testing Core Data Imports...")
try:
    from autods.data.ingestion import DataIngestion
    print("  ✓ DataIngestion")
except Exception as e:
    print(f"  ✗ DataIngestion: {e}")
    errors.append(("DataIngestion", str(e)))

# Test 2: Analysis Imports
print("\n[2] Testing Analysis Imports...")
try:
    from autods.analysis.profiler import StatisticalProfiler
    print("  ✓ StatisticalProfiler")
except Exception as e:
    print(f"  ✗ StatisticalProfiler: {e}")
    errors.append(("StatisticalProfiler", str(e)))

# Test 3: Preprocessing Imports
print("\n[3] Testing Preprocessing Imports...")
try:
    from autods.preprocessing.orchestrator import PreprocessingOrchestrator
    print("  ✓ PreprocessingOrchestrator")
except Exception as e:
    print(f"  ✗ PreprocessingOrchestrator: {e}")
    errors.append(("PreprocessingOrchestrator", str(e)))

try:
    from autods.preprocessing.auto_processor import AutoPreprocessor
    print("  ✓ AutoPreprocessor")
except Exception as e:
    print(f"  ✗ AutoPreprocessor: {e}")
    errors.append(("preprocessing.auto_processor", str(e)))

# Test 4: Modeling Imports
print("\n[4] Testing Modeling Imports...")
try:
    from autods.modeling.orchestrator import ModelOrchestrator
    print("  ✓ ModelOrchestrator")
except Exception as e:
    print(f"  ✗ ModelOrchestrator: {e}")
    errors.append(("ModelOrchestrator", str(e)))

try:
    from autods.modeling.task_detector import TaskDetector
    print("  ✓ TaskDetector")
except Exception as e:
    print(f"  ✗ TaskDetector: {e}")
    errors.append(("TaskDetector", str(e)))

# Test 5: Reports Imports
print("\n[5] Testing Reports Imports...")
try:
    from autods.reports.pipeline import run_complete_analysis
    print("  ✓ run_complete_analysis")
except Exception as e:
    print(f"  ✗ run_complete_analysis: {e}")
    errors.append(("run_complete_analysis", str(e)))

# Test 6: Utils Imports
print("\n[6] Testing Utils Imports...")
try:
    from autods.utils.error_handler import ErrorHandler
    print("  ✓ ErrorHandler")
except Exception as e:
    print(f"  ✗ ErrorHandler: {e}")
    errors.append(("ErrorHandler", str(e)))

try:
    from autods.utils.session_manager import SessionManager, auto_save_session
    print("  ✓ SessionManager, auto_save_session")
except Exception as e:
    print(f"  ✗ session_manager: {e}")
    errors.append(("session_manager", str(e)))

try:
    from autods.utils.checklist import AnalysisChecklist, get_checklist
    print("  ✓ AnalysisChecklist, get_checklist")
except Exception as e:
    print(f"  ✗ checklist: {e}")
    errors.append(("checklist", str(e)))

try:
    from autods.utils.export_generator import ExportGenerator, generate_analysis_exports
    print("  ✓ ExportGenerator, generate_analysis_exports")
except Exception as e:
    print(f"  ✗ export_generator: {e}")
    errors.append(("export_generator", str(e)))

try:
    from autods.utils.demo_data import DemoDatasetGenerator, load_demo_dataset
    print("  ✓ DemoDatasetGenerator, load_demo_dataset")
except Exception as e:
    print(f"  ✗ demo_data: {e}")
    errors.append(("demo_data", str(e)))

# Test 7: Core Analyst
print("\n[7] Testing Core Analyst...")
try:
    from autods.core.analyst import AutoDS
    print("  ✓ AutoDS")
except Exception as e:
    print(f"  ✗ AutoDS: {e}")
    errors.append(("AutoDS", str(e)))

try:
    from autods.core.data_profile import DataProfile
    print("  ✓ DataProfile")
except Exception as e:
    print(f"  ✗ DataProfile: {e}")
    errors.append(("DataProfile", str(e)))

# Summary
print("\n" + "=" * 60)
if errors:
    print(f"❌ FAILED: {len(errors)} import errors")
    for name, err in errors:
        print(f"  - {name}: {err}")
    sys.exit(1)
else:
    print("✅ SUCCESS: All imports working!")
    sys.exit(0)
