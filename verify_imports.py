#!/usr/bin/env python3
"""Quick import verification"""
import sys
sys.path.insert(0, 'src')

modules_to_test = [
    ('autods.data.ingestion', 'DataIngestion'),
    ('autods.analysis.profiler', 'StatisticalProfiler'),
    ('autods.preprocessing.orchestrator', 'PreprocessingOrchestrator'),
    ('autods.modeling.orchestrator', 'ModelOrchestrator'),
    ('autods.reports.pipeline', 'run_complete_analysis'),
    ('autods.utils.error_handler', 'ErrorHandler'),
    ('autods.utils.session_manager', 'SessionManager'),
    ('autods.utils.checklist', 'AnalysisChecklist'),
    ('autods.utils.export_generator', 'ExportGenerator'),
    ('autods.utils.demo_data', 'DemoDatasetGenerator'),
]

errors = []
for module, name in modules_to_test:
    try:
        exec(f"from {module} import {name}")
        print(f"✓ {name}")
    except Exception as e:
        print(f"✗ {name}: {e}")
        errors.append((module, name, str(e)))

if errors:
    print(f"\n❌ {len(errors)} errors")
    sys.exit(1)
else:
    print("\n✅ All imports OK")
    sys.exit(0)
