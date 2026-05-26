#!/usr/bin/env python3
from pathlib import Path

base = Path(__file__).parent.parent
dirs = [
    "analysis",
    "analysis/phase1_eda",
    "analysis/phase2_univariate", 
    "analysis/phase3_blocks",
    "analysis/phase4_regularized",
    "analysis/phase5_interpret",
    "analysis/phase6_sensitivity",
    "scripts",
    "src/phases",
]

for d in dirs:
    Path(base / d).mkdir(parents=True, exist_ok=True)
    print(f"Created: {d}")

print("Done!")
