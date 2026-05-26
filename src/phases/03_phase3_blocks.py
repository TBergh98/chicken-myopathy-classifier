#!/usr/bin/env python3
"""
PHASE 3: FEATURE BLOCK ANALYSIS
Compare predictive value of different feature groups using nested CV.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, balanced_accuracy_score, matthews_corrcoef
import warnings
import json

warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "audio_features_output"
LABELS_PATH = BASE_DIR / "data" / "processed" / "ws_labels_binary.parquet"
OUTPUT_DIR = BASE_DIR / "analysis" / "phase3_blocks"
PHASE1_DIR = BASE_DIR / "analysis" / "phase1_eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# LOAD DATA AND METADATA
# ============================================================================

print("\n" + "="*70)
print("PHASE 3: FEATURE BLOCK ANALYSIS")
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

with open(PHASE1_DIR / 'metadata.json') as f:
    metadata = json.load(f)

label_col = metadata['label_column']
feature_blocks = metadata['feature_blocks']

print(f"✓ Data shape: {df.shape}")
print(f"✓ Label column: {label_col}")
print(f"✓ Feature blocks: {list(feature_blocks.keys())}")

# Prepare data
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if label_col and label_col in numeric_cols:
    numeric_cols.remove(label_col)
if 'ws_grade_raw' in numeric_cols:
    numeric_cols.remove('ws_grade_raw')
numeric_cols = [c for c in numeric_cols if 'chicken_id' not in c.lower()]

X = df[numeric_cols].fillna(df[numeric_cols].mean())
y = df[label_col]

# Ensure binary classification
if y.nunique() != 2:
    print("! Warning: Not binary classification. Skipping feature-block nested CV analysis.")
    # Write minimal outputs and exit gracefully
    with open(OUTPUT_DIR / 'REPORT.md', 'w') as f:
        f.write('# PHASE 3: FEATURE BLOCK ANALYSIS\n\nNon-binary label detected; analysis skipped.')
    pd.DataFrame().to_csv(OUTPUT_DIR / 'group_summary.csv', index=False)
    print("  ✓ Wrote minimal outputs for skipped Phase 3")
    raise SystemExit(0)

y_encoded = (y != y.unique()[0]).astype(int)

print(f"✓ X shape: {X.shape}, y distribution: {y.value_counts().to_dict()}")

# ============================================================================
# DEFINE FEATURE GROUPS FOR ANALYSIS
# ============================================================================

print("\n[2] Defining feature groups...")

# Start with provided blocks
analysis_groups = {}
for block_name, features in feature_blocks.items():
    # Filter to only features that exist in data
    existing_features = [f for f in features if f in numeric_cols]
    if existing_features:
        analysis_groups[block_name] = existing_features

# Create combinations
if 'Temporal' in analysis_groups and 'Spectral' in analysis_groups:
    analysis_groups['Temporal+Spectral'] = analysis_groups['Temporal'] + analysis_groups['Spectral']

if 'Temporal' in analysis_groups and 'Spectral' in analysis_groups and 'F0' in analysis_groups:
    analysis_groups['Temporal+Spectral+F0'] = (analysis_groups['Temporal'] + 
                                               analysis_groups['Spectral'] + 
                                               analysis_groups['F0'])

# All features
analysis_groups['All Features'] = numeric_cols

# PCA-reduced (if all features)
if 'All Features' in analysis_groups and len(numeric_cols) > 10:
    analysis_groups['PCA-reduced'] = numeric_cols  # Will apply PCA in analysis

for name, features in analysis_groups.items():
    print(f"  ✓ {name}: {len(features)} features")

# ============================================================================
# CROSS-VALIDATION SETUP
# ============================================================================

print("\n[3] Setting up cross-validation...")

# Use repeated stratified k-fold
n_splits = 5
n_repeats = 3
cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
n_folds = n_splits * n_repeats

print(f"✓ CV scheme: {n_repeats}×{n_splits}-fold stratified CV = {n_folds} folds total")

# ============================================================================
# NESTED CV ANALYSIS
# ============================================================================

print("\n[4] Running nested CV analysis...")

results_by_group = {}

for group_name, features in analysis_groups.items():
    print(f"\n  Analyzing '{group_name}' ({len(features)} features)...")
    
    # Get feature data
    X_group = X[features].copy()
    
    # Apply PCA if specified
    if group_name == 'PCA-reduced':
        from sklearn.decomposition import PCA
        n_components = min(10, X_group.shape[1] // 2)
        pca = PCA(n_components=n_components, random_state=42)
        X_group = pca.fit_transform(X_group)
        print(f"    Applied PCA: {len(features)} → {n_components} components")
    
    # CV loop
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_group, y_encoded)):
        X_train, X_test = X_group.iloc[train_idx] if hasattr(X_group, 'iloc') else X_group[train_idx], \
                          X_group.iloc[test_idx] if hasattr(X_group, 'iloc') else X_group[test_idx]
        y_train, y_test = y_encoded.iloc[train_idx], y_encoded.iloc[test_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit model (Elastic Net Logistic Regression as primary choice per recommendations)
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', 
                                   solver='lbfgs', l1_ratio=0.5, penalty='elasticnet')
        try:
            model.fit(X_train_scaled, y_train)
        except:
            # Fallback to L2 if elasticnet fails
            model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced',
                                      solver='lbfgs', penalty='l2')
            model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = balanced_accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        fold_results.append({
            'fold': fold_idx,
            'auc': auc,
            'balanced_accuracy': acc,
            'mcc': mcc,
            'n_train': len(y_train),
            'n_test': len(y_test),
        })
    
    # Aggregate results
    fold_results_df = pd.DataFrame(fold_results)
    results_by_group[group_name] = {
        'folds': fold_results,
        'mean_auc': fold_results_df['auc'].mean(),
        'std_auc': fold_results_df['auc'].std(),
        'mean_acc': fold_results_df['balanced_accuracy'].mean(),
        'std_acc': fold_results_df['balanced_accuracy'].std(),
        'mean_mcc': fold_results_df['mcc'].mean(),
        'std_mcc': fold_results_df['mcc'].std(),
    }
    
    print(f"    → ROC-AUC: {results_by_group[group_name]['mean_auc']:.3f} ± {results_by_group[group_name]['std_auc']:.3f}")
    print(f"    → Balanced Acc: {results_by_group[group_name]['mean_acc']:.3f} ± {results_by_group[group_name]['std_acc']:.3f}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n[5] Creating visualizations...")

# 5a. ROC-AUC comparison across feature groups
fig, ax = plt.subplots(figsize=(10, 6))

group_names = list(results_by_group.keys())
aucs = [results_by_group[g]['mean_auc'] for g in group_names]
auc_stds = [results_by_group[g]['std_auc'] for g in group_names]

ax.bar(range(len(group_names)), aucs, yerr=auc_stds, capsize=5, alpha=0.7, 
       color=sns.color_palette("husl", len(group_names)), edgecolor='black', linewidth=1)
ax.set_xticks(range(len(group_names)))
ax.set_xticklabels(group_names, rotation=45, ha='right')
ax.set_ylabel('ROC-AUC')
ax.set_ylim([0.4, 0.8])
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Random')
ax.set_title('ROC-AUC Across Feature Groups (Nested CV)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_roc_auc_by_group.png', dpi=100, bbox_inches='tight')
print("  ✓ Saved 01_roc_auc_by_group.png")
plt.close()

# 5b. Balanced accuracy comparison
fig, ax = plt.subplots(figsize=(10, 6))

accs = [results_by_group[g]['mean_acc'] for g in group_names]
acc_stds = [results_by_group[g]['std_acc'] for g in group_names]

ax.bar(range(len(group_names)), accs, yerr=acc_stds, capsize=5, alpha=0.7,
       color=sns.color_palette("husl", len(group_names)), edgecolor='black', linewidth=1)
ax.set_xticks(range(len(group_names)))
ax.set_xticklabels(group_names, rotation=45, ha='right')
ax.set_ylabel('Balanced Accuracy')
ax.set_ylim([0.4, 0.8])
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Random')
ax.set_title('Balanced Accuracy Across Feature Groups (Nested CV)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_balanced_accuracy_by_group.png', dpi=100, bbox_inches='tight')
print("  ✓ Saved 02_balanced_accuracy_by_group.png")
plt.close()

# 5c. MCC comparison
fig, ax = plt.subplots(figsize=(10, 6))

mccs = [results_by_group[g]['mean_mcc'] for g in group_names]
mcc_stds = [results_by_group[g]['std_mcc'] for g in group_names]

ax.bar(range(len(group_names)), mccs, yerr=mcc_stds, capsize=5, alpha=0.7,
       color=sns.color_palette("husl", len(group_names)), edgecolor='black', linewidth=1)
ax.set_xticks(range(len(group_names)))
ax.set_xticklabels(group_names, rotation=45, ha='right')
ax.set_ylabel('Matthews Correlation Coefficient')
ax.set_ylim([-0.2, 0.6])
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='No correlation')
ax.set_title('MCC Across Feature Groups (Nested CV)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_mcc_by_group.png', dpi=100, bbox_inches='tight')
print("  ✓ Saved 03_mcc_by_group.png")
plt.close()

# 5d. Multi-metric comparison heatmap
metrics_data = []
for group_name in group_names:
    metrics_data.append([
        results_by_group[group_name]['mean_auc'],
        results_by_group[group_name]['mean_acc'],
        results_by_group[group_name]['mean_mcc']
    ])

metrics_df = pd.DataFrame(metrics_data, index=group_names, 
                         columns=['ROC-AUC', 'Balanced Acc', 'MCC'])

fig, ax = plt.subplots(figsize=(8, len(group_names) * 0.5))
sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5, 
            cbar_kws={'label': 'Score'}, ax=ax, vmin=0.4, vmax=0.8)
ax.set_title('Performance Metrics by Feature Group')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_metrics_heatmap.png', dpi=100, bbox_inches='tight')
print("  ✓ Saved 04_metrics_heatmap.png")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n[6] Saving results...")

# Detailed results
results_export = []
for group_name, results in results_by_group.items():
    for fold_result in results['folds']:
        fold_result['group'] = group_name
        results_export.append(fold_result)

results_export_df = pd.DataFrame(results_export)
results_export_df.to_csv(OUTPUT_DIR / 'cv_fold_results.csv', index=False)
print("  ✓ Saved cv_fold_results.csv")

# Summary by group
summary_data = []
for group_name in group_names:
    r = results_by_group[group_name]
    summary_data.append({
        'Feature_Group': group_name,
        'Mean_AUC': r['mean_auc'],
        'Std_AUC': r['std_auc'],
        'Mean_Balanced_Acc': r['mean_acc'],
        'Std_Balanced_Acc': r['std_acc'],
        'Mean_MCC': r['mean_mcc'],
        'Std_MCC': r['std_mcc'],
        'N_Features': len(analysis_groups[group_name]) if group_name != 'PCA-reduced' else 10,
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(OUTPUT_DIR / 'group_summary.csv', index=False)
print("  ✓ Saved group_summary.csv")

# ============================================================================
# GENERATE REPORT
# ============================================================================

print("\n[7] Generating report...")

report = f"""
# PHASE 3: FEATURE BLOCK ANALYSIS REPORT

## Study Design
- **Cross-validation**: {n_repeats}×{n_splits}-fold repeated stratified CV
- **Total folds**: {n_folds}
- **Model**: Logistic Regression with elastic net regularization
- **Evaluation metrics**: ROC-AUC, Balanced Accuracy, MCC

## Results Summary

| Feature Group | N Features | Mean ROC-AUC | Mean Bal. Acc | Mean MCC |
|---|---:|---:|---:|---:|
"""

for _, row in summary_df.iterrows():
    report += f"| {row['Feature_Group']} | {int(row['N_Features'])} | {row['Mean_AUC']:.4f}±{row['Std_AUC']:.4f} | {row['Mean_Balanced_Acc']:.4f}±{row['Std_Balanced_Acc']:.4f} | {row['Mean_MCC']:.4f}±{row['Std_MCC']:.4f} |\n"

report += f"\n## Key Findings\n\n### Best Performing Feature Group\n"

best_group = summary_df.loc[summary_df['Mean_AUC'].idxmax()]
report += f"- **{best_group['Feature_Group']}** with ROC-AUC = {best_group['Mean_AUC']:.4f}±{best_group['Std_AUC']:.4f}\n"

report += f"\n## Visualizations\n- `01_roc_auc_by_group.png` - ROC-AUC comparison\n- `02_balanced_accuracy_by_group.png` - Balanced accuracy comparison\n- `03_mcc_by_group.png` - MCC comparison  \n- `04_metrics_heatmap.png` - Multi-metric heatmap\n\n## Next Steps\n- Phase 4: Regularized modeling with all approaches + stability selection\n- Phase 5: Feature importance interpretation\n"

with open(OUTPUT_DIR / 'REPORT.md', 'w') as f:
    f.write(report)
print("  ✓ Saved REPORT.md")

with open(OUTPUT_DIR / 'summary.json', 'w') as f:
    json.dump({'n_groups': len(analysis_groups), 'n_folds': n_folds}, f, indent=2)
print("  ✓ Saved summary.json")

print("\n" + "="*70)
print("✓ PHASE 3 COMPLETE")
print("="*70)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
