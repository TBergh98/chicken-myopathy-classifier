#!/usr/bin/env python3
"""
PHASE 1: EXPLORATORY DATA ANALYSIS
Comprehensive data quality checks, distribution analysis, and correlation structure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================

# Robust workspace root detection (works when this script is in src/phases/)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "audio_features_output"
LABELS_PATH = BASE_DIR / "data" / "processed" / "ws_labels_binary.parquet"
OUTPUT_DIR = BASE_DIR / "analysis" / "phase1_eda"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n" + "="*70)
print("PHASE 1: EXPLORATORY DATA ANALYSIS")
print("="*70)

print("\n[1] Loading data...")
df = pd.read_parquet(DATA_DIR / "audio_features.parquet")
print(f"✓ Loaded data shape: {df.shape}")
print(f"✓ Columns: {df.columns.tolist()[:5]}... ({len(df.columns)} total)")

if not LABELS_PATH.exists():
    raise FileNotFoundError(
        f"Binary label table not found: {LABELS_PATH}. Run scripts/prepare_ws_binary_labels.py first."
    )

labels_df = pd.read_parquet(LABELS_PATH)
df = df.merge(labels_df, on="chicken_id", how="left", validate="m:1")
print(f"✓ Loaded labels: {labels_df.shape}")
print(f"✓ Matched labeled rows: {df['ws_myopathy_binary'].notna().sum()}/{len(df)}")

df = df.dropna(subset=["ws_myopathy_binary"]).copy()
df["ws_myopathy_binary"] = df["ws_myopathy_binary"].astype(int)
df["ws_grade_raw"] = pd.to_numeric(df["ws_grade_raw"], errors="coerce")
print(f"✓ Analysis rows after label join: {df.shape}")

# ============================================================================
# 2. BASIC DATA QUALITY
# ============================================================================

print("\n[2] Data Quality Check...")

# Examine columns and types
print(f"\nData types distribution:")
print(df.dtypes.value_counts())

# Check for identifier columns
id_cols = [c for c in df.columns if 'chicken_id' in c.lower() or 'id' in c.lower()]
print(f"\nIdentifier columns found: {id_cols}")

# Check for missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print(f"\nMissing values:")
    print(missing[missing > 0])
else:
    print("\n✓ No missing values")

# Check for duplicate rows
dupes = df.duplicated().sum()
print(f"✓ Duplicate rows: {dupes}")

analysis_excluded_columns = {"ws_grade_raw"}

# ============================================================================
# 3. CLASS BALANCE (assuming 'myopathy' or similar label exists)
# ============================================================================

print("\n[3] Looking for target/label column...")

# Search for common label column names
label_candidates = ['myopathy', 'disease', 'condition', 'status', 'health', 
                     'label', 'target', 'class']
label_col = None
for candidate in label_candidates:
    matching = [c for c in df.columns if candidate.lower() in c.lower()]
    if matching:
        label_col = matching[0]
        print(f"✓ Found label column: '{label_col}'")
        break

if label_col is None:
    print("! Warning: No clear label column found. Examining non-numeric columns...")
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  Potential label columns: {non_numeric}")
        label_col = non_numeric[0]
        print(f"  Using: {label_col}")

if label_col:
    print(f"\nClass distribution:")
    print(df[label_col].value_counts())
    print(f"\nClass balance:")
    print(df[label_col].value_counts(normalize=True))
    
    # Store label info
    label_info = {
        'column': label_col,
        'classes': df[label_col].nunique(),
        'distribution': df[label_col].value_counts().to_dict()
    }
else:
    print("! No label column identified - treating as exploration only")
    label_info = None

# ============================================================================
# 4. FEATURE STRUCTURE
# ============================================================================

print("\n[4] Feature Structure...")

# Identify feature blocks
feature_blocks = {}

# Temporal features
temporal_features = [c for c in df.columns if any(x in c.lower() for x in 
                     ['silence', 'vocalization', 'event', 'duration', 'rate'])]
if temporal_features:
    feature_blocks['Temporal'] = temporal_features
    print(f"✓ Temporal features ({len(temporal_features)}): {temporal_features[:3]}...")

# Energy features
energy_features = [c for c in df.columns if any(x in c.lower() for x in 
                   ['rms', 'energy', 'amplitude'])]
if energy_features:
    feature_blocks['Energy'] = energy_features
    print(f"✓ Energy features ({len(energy_features)}): {energy_features}")

# Spectral features
spectral_features = [c for c in df.columns if any(x in c.lower() for x in 
                     ['spectral', 'zcr', 'freq', 'hz'])]
if spectral_features:
    feature_blocks['Spectral'] = spectral_features
    print(f"✓ Spectral features ({len(spectral_features)}): {spectral_features[:3]}...")

# F0/Pitch features
f0_features = [c for c in df.columns if any(x in c.lower() for x in 
               ['f0', 'pitch', 'fundamental'])]
if f0_features:
    feature_blocks['F0'] = f0_features
    print(f"✓ F0/Pitch features ({len(f0_features)}): {f0_features}")

# MFCC features
mfcc_features = [c for c in df.columns if 'mfcc' in c.lower()]
if mfcc_features:
    feature_blocks['MFCC'] = mfcc_features
    print(f"✓ MFCC features ({len(mfcc_features)}): {mfcc_features[:3]}...")

# Other features
all_feature_cols = set()
for features in feature_blocks.values():
    all_feature_cols.update(features)
other_features = [c for c in df.columns if c not in all_feature_cols and 
                  c not in id_cols and c != label_col and c not in analysis_excluded_columns]
if other_features:
    feature_blocks['Other'] = other_features
    print(f"✓ Other features ({len(other_features)}): {other_features}")

print(f"\nTotal feature blocks: {len(feature_blocks)}")
print(f"Total numeric features: {sum(len(v) for v in feature_blocks.values())}")

# ============================================================================
# 5. DISTRIBUTION ANALYSIS
# ============================================================================

print("\n[5] Feature Distribution Analysis...")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if label_col and label_col in numeric_cols:
    numeric_cols.remove(label_col)
if 'ws_grade_raw' in numeric_cols:
    numeric_cols.remove('ws_grade_raw')
if id_cols:
    numeric_cols = [c for c in numeric_cols if c not in id_cols]

# Basic statistics
print(f"\nBasic statistics (numeric features):")
stats = df[numeric_cols].describe().T
print(stats[['mean', 'std', 'min', 'max']])

# ============================================================================
# 6. OUTLIER DETECTION (simple z-score)
# ============================================================================

print("\n[6] Outlier Detection (Z-score > 3)...")

outlier_counts = {}
for col in numeric_cols:
    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
    outlier_count = (z_scores > 3).sum()
    if outlier_count > 0:
        outlier_counts[col] = outlier_count

if outlier_counts:
    print(f"Features with outliers (Z > 3):")
    for col, count in sorted(outlier_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {col}: {count} outliers")
else:
    print("✓ No extreme outliers detected (Z > 3)")

# ============================================================================
# 7. CORRELATION ANALYSIS
# ============================================================================

print("\n[7] Correlation Analysis...")

# Compute correlation matrix
corr_matrix = df[numeric_cols].corr(method='spearman')

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.85:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

high_corr_pairs.sort(key=lambda x: -abs(x[2]))
print(f"\nHighly correlated pairs (|r| > 0.85): {len(high_corr_pairs)}")
if high_corr_pairs:
    for feat1, feat2, corr in high_corr_pairs[:10]:
        print(f"  {feat1} <-> {feat2}: {corr:.3f}")

# ============================================================================
# 8. CREATE VISUALIZATIONS
# ============================================================================

print("\n[8] Creating visualizations...")

# 8a. Class balance pie chart (if label exists)
if label_col:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    label_counts = df[label_col].value_counts()
    axes[0].pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0].set_title(f'Class Distribution\n(n={len(df)})')
    
    axes[1].bar(range(len(label_counts)), label_counts.values)
    axes[1].set_xticks(range(len(label_counts)))
    axes[1].set_xticklabels(label_counts.index)
    axes[1].set_ylabel('Count')
    axes[1].set_title('Class Counts')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_class_distribution.png', dpi=100, bbox_inches='tight')
    print("  ✓ Saved 01_class_distribution.png")
    plt.close()

# 8b. Correlation heatmap (subset for readability)
# Select top features by variance
top_n = min(30, len(numeric_cols))
top_features = df[numeric_cols].var().nlargest(top_n).index
corr_subset = df[top_features].corr(method='spearman')

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_subset, cmap='coolwarm', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title(f'Correlation Heatmap\n(Top {top_n} features by variance)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_correlation_heatmap.png', dpi=100, bbox_inches='tight')
print("  ✓ Saved 02_correlation_heatmap.png")
plt.close()

# 8c. Feature distribution by class (if label exists)
if label_col:
    # Select 8 most variable features
    top_features_list = df[numeric_cols].var().nlargest(8).index
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features_list):
        for label_val in df[label_col].unique():
            data = df[df[label_col] == label_val][feature]
            axes[idx].hist(data, alpha=0.6, label=f'{label_col}={label_val}', bins=15)
        axes[idx].set_title(feature)
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_top_features_by_class.png', dpi=100, bbox_inches='tight')
    print("  ✓ Saved 03_top_features_by_class.png")
    plt.close()

# 8d. Feature block boxplot
if label_col:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    block_names = list(feature_blocks.keys())[:6]
    
    for idx, block_name in enumerate(block_names):
        features = feature_blocks[block_name]
        # Normalize and aggregate
        block_data = df[features].copy()
        block_data_norm = (block_data - block_data.mean()) / (block_data.std() + 1e-8)
        block_mean = block_data_norm.mean(axis=1)
        
        plot_df = pd.DataFrame({
            'Score': block_mean,
            label_col: df[label_col]
        })
        
        sns.boxplot(data=plot_df, x=label_col, y='Score', ax=axes[idx])
        axes[idx].set_title(f'{block_name} Block\n({len(features)} features)')
    
    # Hide unused subplots
    for idx in range(len(block_names), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_feature_blocks_by_class.png', dpi=100, bbox_inches='tight')
    print("  ✓ Saved 04_feature_blocks_by_class.png")
    plt.close()

# 8e. Feature variance plot
fig, ax = plt.subplots(figsize=(12, 6))
variances = df[numeric_cols].var().nlargest(20)
ax.barh(range(len(variances)), variances.values)
ax.set_yticks(range(len(variances)))
ax.set_yticklabels(variances.index)
ax.set_xlabel('Variance')
ax.set_title('Top 20 Features by Variance')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_feature_variance.png', dpi=100, bbox_inches='tight')
print("  ✓ Saved 05_feature_variance.png")
plt.close()

# ============================================================================
# 9. SAVE SUMMARY REPORT
# ============================================================================

print("\n[9] Generating summary report...")

report = f"""
# PHASE 1: EXPLORATORY DATA ANALYSIS REPORT

## Dataset Overview
- **Total samples**: {len(df)}
- **Total features**: {len(numeric_cols)}
- **Feature blocks**: {len(feature_blocks)}

## Feature Blocks
"""

for block_name, features in feature_blocks.items():
    report += f"\n### {block_name} ({len(features)} features)\n"
    report += f"{', '.join(features[:5])}"
    if len(features) > 5:
        report += f", ... (+{len(features)-5} more)"
    report += "\n"

if label_col:
    report += f"\n## Target Variable: {label_col}\n- **Classes**: {label_info['classes']}\n- **Distribution**: \n"
    for cls, count in label_info['distribution'].items():
        pct = 100 * count / len(df)
        report += f"  - {cls}: {count} ({pct:.1f}%)\n"

report += f"\n\n## Data Quality\n- **Missing values**: {missing.sum() if missing.sum() > 0 else 0}\n- **Duplicate rows**: {dupes}\n- **Outliers (Z > 3)**: {len(outlier_counts)} features with outliers\n\n## High Correlation Pairs (|r| > 0.85)\n- **Count**: {len(high_corr_pairs)}\n"

if high_corr_pairs:
    report += "\nTop 10 correlated pairs:\n"
    for feat1, feat2, corr in high_corr_pairs[:10]:
        report += f"- {feat1} <-> {feat2}: {corr:.3f}\n"

report += "\n\n## Visualizations Created\n1. `01_class_distribution.png` - Class balance and counts\n2. `02_correlation_heatmap.png` - Correlation structure\n3. `03_top_features_by_class.png` - Feature distributions by class\n4. `04_feature_blocks_by_class.png` - Block-level comparison\n5. `05_feature_variance.png` - Top 20 features by variance\n\n## Next Steps\n- Phase 2: Univariate statistical tests\n- Phase 3: Feature block analysis\n- Phase 4: Regularized modeling with feature selection"

with open(OUTPUT_DIR / 'REPORT.md', 'w') as f:
    f.write(report)
print("  ✓ Saved REPORT.md")

# Save feature block metadata for downstream phases
metadata = {
    'n_samples': len(df),
    'n_features': len(numeric_cols),
    'label_column': label_col,
    'label_source': str(LABELS_PATH),
    'feature_blocks': {k: v for k, v in feature_blocks.items()},
    'high_correlation_pairs': [(f1, f2, float(c)) for f1, f2, c in high_corr_pairs],
}

with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("  ✓ Saved metadata.json")

print("\n" + "="*70)
print("✓ PHASE 1 COMPLETE")
print("="*70)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
