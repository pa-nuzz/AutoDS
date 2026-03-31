#!/usr/bin/env python3
"""Comprehensive AutoDS Test Suite"""
import sys
sys.path.insert(0, 'src')

print("=" * 60)
print("AUTODS COMPREHENSIVE TEST SUITE")
print("=" * 60)

errors = []
warnings = []

# Test 1: Core Data Module
print("\n[1/10] Testing Core Data Module...")
try:
    from autods.data.ingestion import DataIngestion
    print("  ✓ DataIngestion")
except Exception as e:
    print(f"  ✗ DataIngestion: {e}")
    errors.append(("data.ingestion", str(e)))

# Test 2: Analysis Module
print("\n[2/10] Testing Analysis Module...")
try:
    from autods.analysis.profiler import StatisticalProfiler
    print("  ✓ StatisticalProfiler")
except Exception as e:
    print(f"  ✗ StatisticalProfiler: {e}")
    errors.append(("analysis.profiler", str(e)))

# Test 3: Preprocessing Module
print("\n[3/10] Testing Preprocessing Module...")
try:
    from autods.preprocessing.orchestrator import PreprocessingOrchestrator
    print("  ✓ PreprocessingOrchestrator")
except Exception as e:
    print(f"  ✗ PreprocessingOrchestrator: {e}")
    errors.append(("preprocessing.orchestrator", str(e)))

try:
    from autods.preprocessing.auto_processor import AutoPreprocessor
    print("  ✓ AutoPreprocessor")
except Exception as e:
    print(f"  ✗ AutoPreprocessor: {e}")
    errors.append(("preprocessing.auto_processor", str(e)))

# Test 4: Modeling Module
print("\n[4/10] Testing Modeling Module...")
try:
    from autods.modeling.orchestrator import ModelOrchestrator
    print("  ✓ ModelOrchestrator")
except Exception as e:
    print(f"  ✗ ModelOrchestrator: {e}")
    errors.append(("modeling.orchestrator", str(e)))

try:
    from autods.modeling.task_detector import TaskDetector
    print("  ✓ TaskDetector")
except Exception as e:
    print(f"  ✗ TaskDetector: {e}")
    errors.append(("modeling.task_detector", str(e)))

# Test 5: Reports Module
print("\n[5/10] Testing Reports Module...")
try:
    from autods.reports.pipeline import run_complete_analysis
    print("  ✓ run_complete_analysis")
except Exception as e:
    print(f"  ✗ run_complete_analysis: {e}")
    errors.append(("reports.pipeline", str(e)))

# Test 6: Utils - Error Handler
print("\n[6/10] Testing Utils - Error Handler...")
try:
    from autods.utils.error_handler import ErrorHandler
    print("  ✓ ErrorHandler")
except Exception as e:
    print(f"  ✗ ErrorHandler: {e}")
    errors.append(("utils.error_handler", str(e)))

# Test 7: Utils - Session Manager
print("\n[7/10] Testing Utils - Session Manager...")
try:
    from autods.utils.session_manager import SessionManager, auto_save_session
    print("  ✓ SessionManager, auto_save_session")
except Exception as e:
    print(f"  ✗ session_manager: {e}")
    errors.append(("utils.session_manager", str(e)))

# Test 8: Utils - Checklist
print("\n[8/10] Testing Utils - Checklist...")
try:
    from autods.utils.checklist import AnalysisChecklist, get_checklist
    print("  ✓ AnalysisChecklist, get_checklist")
except Exception as e:
    print(f"  ✗ checklist: {e}")
    errors.append(("utils.checklist", str(e)))

# Test 9: Utils - Export Generator
print("\n[9/10] Testing Utils - Export Generator...")
try:
    from autods.utils.export_generator import ExportGenerator, generate_analysis_exports
    print("  ✓ ExportGenerator, generate_analysis_exports")
except Exception as e:
    print(f"  ✗ export_generator: {e}")
    errors.append(("utils.export_generator", str(e)))

# Test 10: Utils - Demo Data
print("\n[10/10] Testing Utils - Demo Data...")
try:
    from autods.utils.demo_data import DemoDatasetGenerator
    print("  ✓ DemoDatasetGenerator")
except Exception as e:
    print(f"  ✗ demo_data: {e}")
    errors.append(("utils.demo_data", str(e)))

# Summary
print("\n" + "=" * 60)
if errors:
    print(f"❌ FAILED: {len(errors)} import errors")
    for module, err in errors:
        print(f"  - {module}: {err}")
    print(f"\nStatus: PROJECT HAS IMPORT ERRORS - NEEDS FIXING")
    sys.exit(1)
else:
    print("✅ SUCCESS: All {0} modules imported successfully!".format(10))
    print("\nStatus: PROJECT IMPORTS WORKING ✓")
    print("\nYou can now run: streamlit run app.py")
    sys.exit(0)
