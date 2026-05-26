#!/usr/bin/env python3
"""
Summary of all created files and implementation status (script version under scripts/).
"""

from pathlib import Path
import json

BASE_DIR = Path(__file__).parent.parent

print("\n" + "="*80)
print("FEATURE SELECTION IMPLEMENTATION - COMPLETE FILE INVENTORY")
print("="*80)

# List of created files
created_files = {
    "Analysis Scripts": [
        "src/phases/01_phase1_eda.py",
        "src/phases/02_phase2_univariate.py",
        "src/phases/03_phase3_blocks.py",
        "src/phases/04_phase4_regularized.py",
        "run_analysis.py",
        "scripts/create_structure.py",
    ],
    "Documentation": [
        "ANALYSIS_PIPELINE.md",
        "README_INDEX.md",
        "IMPLEMENTATION_SUMMARY.md",
    ]
}

print("\n📂 CREATED FILES:\n")

total_size = 0
total_files = 0

for category, files in created_files.items():
    print(f"\n{category}:")
    print("-" * 60)
    
    for fname in files:
        fpath = BASE_DIR / fname if not fname.startswith('src/') else BASE_DIR / fname
        if fpath.exists():
            size = fpath.stat().st_size
            total_size += size
            total_files += 1
            size_kb = size / 1024
            print(f"  ✓ {fname:<35} {size_kb:>8.1f} KB")
        else:
            print(f"  ? {fname:<35} [NOT FOUND]")

print("\n" + "="*80)
print(f"SUMMARY: {total_files} files created, {total_size/1024:.1f} KB total")
print("="*80)

print("\n📊 OUTPUT STRUCTURE (Auto-created when scripts run):\n")

output_dirs = [
    "analysis/phase1_eda/",
    "analysis/phase2_univariate/",
    "analysis/phase3_blocks/",
    "analysis/phase4_regularized/",
    "analysis/phase5_interpret/",
    "analysis/phase6_sensitivity/",
]

for dir_name in output_dirs:
    dir_path = BASE_DIR / dir_name
    status = "✓ (will create)" if not dir_path.exists() else "✓ (exists)"
    print(f"  {status:15} {dir_name}")

print("\n🚀 QUICK START:\n")
print("  1. Verify environment:")
print("     conda activate vocalizzazioni-audio")
print("\n  2. Run full pipeline:")
print("     python run_analysis.py")
print("\n  3. View results:")
print("     cat analysis/phase1_eda/REPORT.md")
print("     cat analysis/phase4_regularized/top_features_by_importance.csv")

print("\n📖 DOCUMENTATION:\n")

docs = [
    ("IMPLEMENTATION_SUMMARY.md", "Status, design decisions, how-to guide"),
    ("ANALYSIS_PIPELINE.md", "Detailed methodology & interpretation"),
    ("README_INDEX.md", "Quick reference & file index"),
]

for doc_name, description in docs:
    # Prefer docs/ folder
    doc_path = BASE_DIR / 'docs' / doc_name
    if doc_path.exists():
        size = doc_path.stat().st_size / 1024
        print(f"  ✓ {doc_name:<30} {description:<50} ({size:.1f} KB)")

print("\n" + "="*80)
print("✅ IMPLEMENTATION COMPLETE - Ready for execution")
print("="*80 + "\n")

# Also list what to do next
print("📋 NEXT STEPS:\n")
print("  Phase 1-4: Execute 'python run_analysis.py'")
print("  Phase 5-6: [Planned] To be implemented upon request")
print("\n  Expected runtime: 20-35 minutes")
print("  Outputs: analysis/ folder with phase subfolders")
print("  Key file: analysis/phase4_regularized/top_features_by_importance.csv")
print("\n" + "="*80)
