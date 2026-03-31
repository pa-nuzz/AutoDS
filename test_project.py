#!/usr/bin/env python3
"""AutoDS Project Test Suite"""
import sys
sys.path.insert(0, '.')

print("=== AutoDS Project Test Suite ===\n")

errors = []

# Test 1: Upload component
print("Test 1: Component Imports")
try:
    from app.components import upload
    print("  ✓ upload")
except Exception as e:
    print(f"  ✗ upload: {e}")
    errors.append(str(e))

try:
    from app.components import insight
    print("  ✓ insight")
except Exception as e:
    print(f"  ✗ insight: {e}")
    errors.append(str(e))

try:
    from app.components import mode_selector
    print("  ✓ mode_selector")
except Exception as e:
    print(f"  ✗ mode_selector: {e}")
    errors.append(str(e))

try:
    from app.components import auto_flow
    print("  ✓ auto_flow")
except Exception as e:
    print(f"  ✗ auto_flow: {e}")
    errors.append(str(e))

try:
    from app.components import guide_flow
    print("  ✓ guide_flow")
except Exception as e:
    print(f"  ✗ guide_flow: {e}")
    errors.append(str(e))

try:
    from app.components import results
    print("  ✓ results")
except Exception as e:
    print(f"  ✗ results: {e}")
    errors.append(str(e))

# Test 2: Core modules
print("\nTest 2: Core Modules")
try:
    from autods.core.analyst import AutoDS
    print("  ✓ AutoDS")
except Exception as e:
    print(f"  ✗ AutoDS: {e}")
    errors.append(str(e))

try:
    from autods.core.data_profile import DataProfile
    print("  ✓ DataProfile")
except Exception as e:
    print(f"  ✗ DataProfile: {e}")
    errors.append(str(e))

# Test 3: Preprocessing
print("\nTest 3: Preprocessing")
try:
    from autods.preprocessing.pipeline import AutoPreprocessor
    print("  ✓ AutoPreprocessor")
except Exception as e:
    print(f"  ✗ AutoPreprocessor: {e}")
    errors.append(str(e))

# Test 4: Modeling
print("\nTest 4: Modeling")
try:
    from autods.modeling.task_detector import TaskDetector
    print("  ✓ TaskDetector")
except Exception as e:
    print(f"  ✗ TaskDetector: {e}")
    errors.append(str(e))

print("\n" + "="*40)
if errors:
    print(f"FAILED: {len(errors)} errors")
    sys.exit(1)
else:
    print("SUCCESS: All tests passed")
    sys.exit(0)
