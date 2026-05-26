#!/usr/bin/env python3
"""
PHASE 2: UNIVARIATE STATISTICS
For each feature: group medians, effect sizes, p-values, FDR correction, single-feature AUC.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
import warnings
import json
from feature_selection_utils import write_feature_manifest

warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================

# Robust workspace root detection
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "audio_features_output"
LABELS_PATH = BASE_DIR / "data" / "processed" / "ws_labels_binary.parquet"
OUTPUT_DIR = BASE_DIR / "analysis" / "phase2_univariate"
PHASE1_DIR = BASE_DIR / "analysis" / "phase1_eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# LOAD DATA AND METADATA
# ============================================================================

print("\n" + "="*70)
print("PHASE 2: UNIVARIATE STATISTICS")
print("="*70)

print("\n[1] Loading data and metadata...")
df = pd.read_parquet(DATA_DIR / "audio_features.parquet")

if not LABELS_PATH.exists():
    raise FileNotFoundError(
        f"Binary label table not found: {LABELS_PATH}. Run scripts/prepare_ws_binary_labels.py first."
    )

labels_df = pd.read_parquet(LABELS_PATH)
df = df.merge(labels_df, on="chicken_id", how="left", validate="m:1")
df = df.dropna(subset=["ws_myopathy_binary"]).copy()
df["ws_myopathy_binary"] = df["ws_myopathy_binary"].astype(int)

# Load metadata from Phase 1
with open(PHASE1_DIR / 'metadata.json') as f:
    metadata = json.load(f)

label_col = metadata['label_column']
print(f"✓ Data shape: {df.shape}")
print(f"✓ Label column: {label_col}")

# ============================================================================
# PREPARE FEATURES
# ============================================================================

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if label_col and label_col in numeric_cols:
    numeric_cols.remove(label_col)
if 'ws_grade_raw' in numeric_cols:
    numeric_cols.remove('ws_grade_raw')

# Remove ID columns
numeric_cols = [c for c in numeric_cols if 'chicken_id' not in c.lower()]

print(f"✓ Analyzing {len(numeric_cols)} numeric features")

# ============================================================================
# UNIVARIATE ANALYSIS
# ============================================================================

print("\n[2] Running univariate statistical tests...")

results = []

for i, feature in enumerate(numeric_cols):
    if (i + 1) % 20 == 0:
        print(f"  Progress: {i+1}/{len(numeric_cols)}")
    
    feature_data = df[feature]
    
    # Skip if too many missing values
    if feature_data.isnull().sum() > len(df) * 0.5:
        continue
    
    feature_data = feature_data.dropna()
    
    # Get label groups
    labels = df.loc[feature_data.index, label_col]
    
    if labels.nunique() != 2:
        print(f"  ! Skipping {feature}: not binary classification")
        continue
    
    class_0 = feature_data[labels == labels.unique()[0]]
    class_1 = feature_data[labels == labels.unique()[1]]
    
    # Skip if either class is empty
    if len(class_0) == 0 or len(class_1) == 0:
        continue
    
    # Group statistics
    median_0 = class_0.median()
    median_1 = class_1.median()
    mean_0 = class_0.mean()
    mean_1 = class_1.mean()
    std_0 = class_0.std()
    std_1 = class_1.std()
    
    # Effect size (Cohen's d)
    cohens_d = (mean_1 - mean_0) / np.sqrt(((len(class_0)-1)*std_0**2 + (len(class_1)-1)*std_1**2) / (len(class_0) + len(class_1) - 2))
    
    # T-test
    t_stat, t_pval = stats.ttest_ind(class_0, class_1)
    
    # Mann-Whitney U test
    u_stat, u_pval = stats.mannwhitneyu(class_0, class_1)
    
    # ROC-AUC (single feature)
    y_true = labels.values
    y_pred = feature_data.values
    try:
        auc_score = roc_auc_score(y_true, y_pred)
    except:
        auc_score = np.nan
    
    results.append({
        'Feature': feature,
        'Mean_Class0': mean_0,
        'Mean_Class1': mean_1,
        'Median_Class0': median_0,
        'Median_Class1': median_1,
        'Std_Class0': std_0,
        'Std_Class1': std_1,
        'CohenD': cohens_d,
        'AbsCohenD': abs(cohens_d),
        'T_pvalue': t_pval,
        'U_pvalue': u_pval,
        'ROC_AUC': auc_score,
        'N_Class0': len(class_0),
        'N_Class1': len(class_1),
    })

print(f"✓ Completed univariate tests for {len(results)} features")

if len(results) == 0:
    print("! No features were tested (likely non-binary or missing label). Writing empty outputs and exiting gracefully.")
    # Write empty outputs to keep pipeline consistent
    pd.DataFrame(columns=[
        'Feature','Mean_Class0','Mean_Class1','Median_Class0','Median_Class1',
        'Std_Class0','Std_Class1','CohenD','AbsCohenD','T_pvalue','U_pvalue','ROC_AUC',
        'N_Class0','N_Class1'
    ]).to_csv(OUTPUT_DIR / 'univariate_results.csv', index=False)
    with open(OUTPUT_DIR / 'REPORT.md', 'w') as f:
        f.write('# PHASE 2: UNIVARIATE STATISTICS\n\nNo binary label found; univariate tests skipped.')
    print("  ✓ Wrote empty univariate_results.csv and REPORT.md")
    # Exit without error
    raise SystemExit(0)

# ============================================================================
# FDR CORRECTION
# ============================================================================

print("\n[3] Applying FDR correction...")

results_df = pd.DataFrame(results)

# Some features can yield undefined test statistics (e.g. constant columns).
# Convert them to non-significant p-values so FDR correction can proceed.
results_df['T_pvalue'] = pd.to_numeric(results_df['T_pvalue'], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0, 1)
results_df['U_pvalue'] = pd.to_numeric(results_df['U_pvalue'], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0, 1)

# FDR correction on t-test p-values
from scipy.stats import false_discovery_control
results_df['T_pvalue_FDR'] = false_discovery_control(results_df['T_pvalue'], method='bh')
results_df['U_pvalue_FDR'] = false_discovery_control(results_df['U_pvalue'], method='bh')

# Significance thresholds
results_df['Significant_T'] = results_df['T_pvalue_FDR'] < 0.05
results_df['Significant_U'] = results_df['U_pvalue_FDR'] < 0.05

print(f"✓ Significant features (T-test, FDR<0.05): {results_df['Significant_T'].sum()}")
print(f"✓ Significant features (U-test, FDR<0.05): {results_df['Significant_U'].sum()}")

# ============================================================================
# RANKING AND SORTING
# ============================================================================

print("\n[4] Ranking features...")

# Sort by ROC-AUC
results_sorted_auc = results_df.sort_values('ROC_AUC', ascending=False)

# Sort by Cohen's d (absolute)
results_sorted_cohend = results_df.sort_values('AbsCohenD', ascending=False)

# Sort by U-test p-value
results_sorted_pval = results_df.sort_values('U_pvalue', ascending=True)

print("\nTop 15 features by ROC-AUC:")
for idx, (_, row) in enumerate(results_sorted_auc.head(15).iterrows(), 1):
    print(f"  {idx:2d}. {row['Feature']:35s} AUC={row['ROC_AUC']:.3f} (d={row['CohenD']:.3f})")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n[5] Creating visualizations...")

# 5a. Volcano plot
fig, ax = plt.subplots(figsize=(12, 8))

x = results_df['CohenD']
y = -np.log10(results_df['U_pvalue_FDR'] + 1e-300)

# Color by significance
colors = ['red' if sig else 'gray' for sig in results_df['Significant_U']]

scatter = ax.scatter(x, y, c=colors, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)

# Add labels for top features
top_features_volcano = results_sorted_cohend.head(10)
for _, row in top_features_volcano.iterrows():
    ax.annotate(row['Feature'], 
                xy=(row['CohenD'], -np.log10(row['U_pvalue_FDR'] + 1e-300)),
                xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

ax.axhline(y=-np.log10(0.05), color='blue', linestyle='--', linewidth=1, alpha=0.5, label='FDR=0.05')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel("Cohen's d (effect size)", fontsize=12)
ax.set_ylabel("-log10(p-value, FDR-corrected)", fontsize=12)
ax.set_title("Volcano Plot: Effect Size vs Statistical Significance", fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_volcano_plot.png', dpi=100, bbox_inches='tight')
print("  ✓ Saved 01_volcano_plot.png")
plt.close()

# 5b. ROC-AUC ranking barplot
fig, ax = plt.subplots(figsize=(10, 12))
top_n = 30
top_auc = results_sorted_auc.head(top_n)
colors_auc = ['darkred' if x < 0.5 else 'darkgreen' for x in top_auc['ROC_AUC']]
ax.barh(range(len(top_auc)), top_auc['ROC_AUC'], color=colors_auc, alpha=0.7)
ax.set_yticks(range(len(top_auc)))
ax.set_yticklabels(top_auc['Feature'])
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('ROC-AUC')
ax.set_title(f'Top {top_n} Features by Single-Feature ROC-AUC')
ax.set_xlim([0.4, 0.7])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_roc_auc_ranking.png', dpi=100, bbox_inches='tight')
print("  ✓ Saved 02_roc_auc_ranking.png")
plt.close()

# 5c. Effect size ranking
fig, ax = plt.subplots(figsize=(10, 12))
top_cohend = results_sorted_cohend.head(top_n)
colors_d = ['darkred' if x < 0 else 'darkblue' for x in top_cohend['CohenD']]
ax.barh(range(len(top_cohend)), top_cohend['CohenD'], color=colors_d, alpha=0.7)
ax.set_yticks(range(len(top_cohend)))
ax.set_yticklabels(top_cohend['Feature'])
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel("Cohen's d")
ax.set_title(f"Top {top_n} Features by Cohen's d (Effect Size)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_cohend_ranking.png', dpi=100, bbox_inches='tight')
print("  ✓ Saved 03_cohend_ranking.png")
plt.close()

# 5d. p-value distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(results_df['T_pvalue'], bins=30, alpha=0.7, edgecolor='black')
axes[0].set_xlabel('P-value (t-test)')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Univariate P-values (T-test)')
axes[0].axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
axes[0].legend()

axes[1].hist(results_df['ROC_AUC'], bins=30, alpha=0.7, edgecolor='black')
axes[1].set_xlabel('ROC-AUC')
axes[1].set_ylabel('Count')
axes[1].set_title('Distribution of Single-Feature ROC-AUC')
axes[1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Random')
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_pvalue_auc_distributions.png', dpi=100, bbox_inches='tight')
print("  ✓ Saved 04_pvalue_auc_distributions.png")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n[6] Saving results...")

# Save full results table
results_export = results_df.copy()
results_export = results_export.sort_values('ROC_AUC', ascending=False)
results_export.to_csv(OUTPUT_DIR / 'univariate_results.csv', index=False)
print("  ✓ Saved univariate_results.csv")

# Save top significant features
top_sig = results_df[results_df['Significant_U']].sort_values('AbsCohenD', ascending=False)
if len(top_sig) > 0:
    top_sig.to_csv(OUTPUT_DIR / 'significant_features.csv', index=False)
    print(f"  ✓ Saved significant_features.csv ({len(top_sig)} features)")

    manifest_features = top_sig['Feature'].tolist()
    write_feature_manifest(
        OUTPUT_DIR / 'feature_manifest_significant.json',
        name='phase2_significant_features',
        source_phase='phase2_univariate',
        selection_rule='U-test FDR<0.05 sorted by absolute Cohen\'s d',
        features=manifest_features,
        metadata={
            'n_features_analyzed': int(len(results_df)),
            'n_significant_u': int(len(top_sig)),
        },
    )
    print(f"  ✓ Saved feature_manifest_significant.json ({len(manifest_features)} features)")

# ============================================================================
# GENERATE REPORT
# ============================================================================

print("\n[7] Generating report...")

report = f"""
# PHASE 2: UNIVARIATE STATISTICS REPORT

## Overview
- **Features analyzed**: {len(results_df)}
- **Significant features (FDR<0.05, U-test)**: {results_df['Significant_U'].sum()}

## Top 20 Features by ROC-AUC

| Rank | Feature | ROC-AUC | Cohen's d | U-test p | Significant |
|------|---------|---------|-----------|----------|-------------|
"""

for idx, (_, row) in enumerate(results_sorted_auc.head(20).iterrows(), 1):
    sig_str = "✓" if row['Significant_U'] else "-"
    report += f"| {idx} | {row['Feature']} | {row['ROC_AUC']:.4f} | {row['CohenD']:.4f} | {row['U_pvalue']:.2e} | {sig_str} |\n"

report += f"\n## Top 20 Features by Effect Size (Cohen's d)\n\n| Rank | Feature | Cohen's d | ROC-AUC | Median Diff |\n|------|---------|-----------|---------|------------|\n"

for idx, (_, row) in enumerate(results_sorted_cohend.head(20).iterrows(), 1):
    median_diff = row['Median_Class1'] - row['Median_Class0']
    report += f"| {idx} | {row['Feature']} | {row['CohenD']:.4f} | {row['ROC_AUC']:.4f} | {median_diff:.4f} |\n"

report += f"\n## Statistical Summary\n- **Features with AUC > 0.55**: {(results_df['ROC_AUC'] > 0.55).sum()}\n- **Features with AUC > 0.60**: {(results_df['ROC_AUC'] > 0.60).sum()}\n- **Features with |Cohen's d| > 0.5**: {(results_df['AbsCohenD'] > 0.5).sum()}\n- **Mean ROC-AUC**: {results_df['ROC_AUC'].mean():.4f}\n- **Median ROC-AUC**: {results_df['ROC_AUC'].median():.4f}\n\n## Visualizations\n- `01_volcano_plot.png` - Effect size vs statistical significance\n- `02_roc_auc_ranking.png` - Single-feature prediction performance\n- `03_cohend_ranking.png` - Feature effect sizes\n- `04_pvalue_auc_distributions.png` - P-value and AUC distributions\n\n## Key Findings\n- Univariate analysis identifies features that independently differ between classes\n- ROC-AUC > 0.55 indicates modest predictive power\n- Significant features warrant inclusion in multivariate models\n- Non-significant features may still contribute to multivariate predictions\n\n## Next Steps\n- Phase 3: Feature block analysis to compare feature group predictive value\n- Phase 4: Regularized modeling with stability selection\n"""

with open(OUTPUT_DIR / 'REPORT.md', 'w', encoding='utf-8') as f:
    f.write(report)
print("  ✓ Saved REPORT.md")

with open(OUTPUT_DIR / 'summary.json', 'w') as f:
    json.dump({'n_features_analyzed': len(results_df), 'n_significant_u': int(results_df['Significant_U'].sum())}, f, indent=2)
print("  ✓ Saved summary.json")

print("\n" + "="*70)
print("✓ PHASE 2 COMPLETE")
print("="*70)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
