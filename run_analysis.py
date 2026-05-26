#!/usr/bin/env python3
"""
MASTER RUNNER: Feature Selection Analysis Pipeline
Executes all phases in sequence with organized output.
"""

import subprocess
import sys
from pathlib import Path
import time

BASE_DIR = Path(__file__).parent

print("\n" + "="*80)
print(" "*20 + "FEATURE SELECTION ANALYSIS PIPELINE")
print("="*80)

# Prepare binary WS labels from the workbook before running the analysis phases.
preprocess_script = BASE_DIR / "scripts" / "prepare_ws_binary_labels.py"
print("\n[Preparing input labels...]")
preprocess_result = subprocess.run(
    [sys.executable, str(preprocess_script)],
    cwd=str(BASE_DIR),
    capture_output=False,
    timeout=3600,
)
if preprocess_result.returncode != 0:
    print("\n✗ Label preprocessing failed. Stop before running analysis phases.")
    sys.exit(preprocess_result.returncode)

# Ensure output directories exist
output_dirs = [
    "analysis/phase1_eda",
    "analysis/phase2_univariate",
    "analysis/phase3_blocks",
    "analysis/phase4_regularized",
    "analysis/phase5_interpret",
    "analysis/phase6_sensitivity",
]

for dir_name in output_dirs:
    Path(BASE_DIR / dir_name).mkdir(parents=True, exist_ok=True)

# Define phases
phases = [
    {
        'number': 1,
        'name': 'Exploratory Data Analysis',
        'script': '01_phase1_eda.py',
        'description': 'Data quality checks, distributions, correlations'
    },
    {
        'number': 2,
        'name': 'Univariate Statistics',
        'script': '02_phase2_univariate.py',
        'description': 'Statistical tests, effect sizes, p-values, FDR correction'
    },
    {
        'number': 3,
        'name': 'Feature Block Analysis',
        'script': '03_phase3_blocks.py',
        'description': 'Compare predictive value of feature groups'
    },
    {
        'number': 4,
        'name': 'Regularized Modelling',
        'script': '04_phase4_regularized.py',
        'description': 'Elastic Net, LASSO, SVM, RF with stability selection'
    },
]

# Run phases
completed = 0
failed = 0

for phase in phases:
    print(f"\n{'='*80}")
    print(f"PHASE {phase['number']}: {phase['name'].upper()}")
    print(f"{'='*80}")
    print(f"Description: {phase['description']}")
    print(f"Script: {phase['script']}")
    
    # Canonical location for phase implementations
    script_path = BASE_DIR / 'src' / 'phases' / phase['script']

    if not script_path.exists():
        print(f"✗ SKIPPED: Script not found at canonical location {script_path}")
        failed += 1
        continue
    
    print("\n[Starting...]")
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
            capture_output=False,
            timeout=3600  # 1 hour timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✓ PHASE {phase['number']} COMPLETED ({elapsed:.1f}s)")
            completed += 1
        else:
            print(f"\n✗ PHASE {phase['number']} FAILED (exit code: {result.returncode})")
            failed += 1
    
    except subprocess.TimeoutExpired:
        print(f"\n✗ PHASE {phase['number']} TIMEOUT (exceeded 1 hour)")
        failed += 1
    except Exception as e:
        print(f"\n✗ PHASE {phase['number']} ERROR: {e}")
        failed += 1

# Final summary
print(f"\n{'='*80}")
print("PIPELINE SUMMARY")
print(f"{'='*80}")
print(f"Completed: {completed}/{len(phases)}")
print(f"Failed: {failed}/{len(phases)}")

if failed == 0:
    print(f"\n✓ ALL PHASES COMPLETED SUCCESSFULLY!")
    print(f"\nResults organized in:")
    for dir_name in output_dirs[:4]:  # Show first 4 phases
        print(f"  - analysis/{dir_name.split('/')[-1]}/")
    print(f"\nNEXT: Run Phase 5 (Interpretation) manually or modify this script to include it.")
else:
    print(f"\n✗ Some phases failed. Review output above.")

print("="*80 + "\n")
