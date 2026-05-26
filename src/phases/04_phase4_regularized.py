#!/usr/bin/env python3
"""
PHASE 4: REGULARIZED MODELLING WITH STABILITY SELECTION
Elastic Net, LASSO, SVM, Random Forest with nested CV and feature selection stability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import time
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef, precision_recall_curve, auc as auc_pr
import warnings
import json

from feature_selection_utils import load_feature_manifest, write_feature_manifest, deduplicate_features

warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "audio_features_output"
LABELS_PATH = BASE_DIR / "data" / "processed" / "ws_labels_binary.parquet"
OUTPUT_DIR = BASE_DIR / "analysis" / "phase4_regularized"
PHASE1_DIR = BASE_DIR / "analysis" / "phase1_eda"
PHASE2_DIR = BASE_DIR / "analysis" / "phase2_univariate"
PHASE3_DIR = BASE_DIR / "analysis" / "phase3_blocks"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

parser = argparse.ArgumentParser(description='Phase 4 regularized modeling')
parser.add_argument(
    '--feature-mode',
    choices=['all', 'reduced'],
    default='reduced',
    help='Feature set to use: all numeric features or the default reduced set.',
)
parser.add_argument(
    '--feature-manifest',
    type=str,
    default=None,
    help='Optional path to a custom feature manifest JSON file.',
)
args = parser.parse_args()

# ============================================================================
# LOAD DATA AND METADATA
# ============================================================================

print("\n" + "="*70)
print("PHASE 4: REGULARIZED MODELLING & STABILITY SELECTION")
print("="*70)

print("\n[1] Loading data...")
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
print(f"✓ Data shape: {df.shape}")
print(f"✓ Label: {label_col}")

# Prepare data
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if label_col and label_col in numeric_cols:
    numeric_cols.remove(label_col)
if 'ws_grade_raw' in numeric_cols:
    numeric_cols.remove('ws_grade_raw')
numeric_cols = [c for c in numeric_cols if 'chicken_id' not in c.lower()]

def _load_reduced_feature_set() -> tuple[list[str], str]:
    phase2_manifest = PHASE2_DIR / 'feature_manifest_significant.json'
    phase3_manifest = PHASE3_DIR / 'feature_manifest_best_block.json'

    if not phase2_manifest.exists() or not phase3_manifest.exists():
        missing = []
        if not phase2_manifest.exists():
            missing.append(str(phase2_manifest))
        if not phase3_manifest.exists():
            missing.append(str(phase3_manifest))
        raise FileNotFoundError(
            'Reduced feature mode requires phase 2 and phase 3 manifests. Missing: ' + ', '.join(missing)
        )

    phase2_features = load_feature_manifest(phase2_manifest)['features']
    phase3_features = load_feature_manifest(phase3_manifest)['features']
    combined = deduplicate_features(list(phase3_features) + list(phase2_features))
    return combined, f"reduced: {phase3_manifest.name} + {phase2_manifest.name}"


if args.feature_manifest:
    manifest_path = Path(args.feature_manifest)
    if not manifest_path.is_absolute():
        manifest_path = BASE_DIR / manifest_path
    manifest = load_feature_manifest(manifest_path)
    selected_features = [f for f in manifest['features'] if f in numeric_cols]
    feature_source = f"custom manifest: {manifest_path}"
elif args.feature_mode == 'reduced':
    selected_features, feature_source = _load_reduced_feature_set()
    selected_features = [f for f in selected_features if f in numeric_cols]
else:
    selected_features = list(numeric_cols)
    feature_source = 'all numeric features'

selected_features = deduplicate_features([f for f in selected_features if f in numeric_cols])
if len(selected_features) == 0:
    raise ValueError('No usable features selected for phase 4 modeling.')

X = df[selected_features].fillna(df[selected_features].mean())
y = df[label_col]
y_encoded = (y != y.unique()[0]).astype(int)

print(f"✓ Feature mode: {args.feature_mode}")
print(f"✓ Feature source: {feature_source}")
print(f"✓ {len(selected_features)} features, {len(X)} samples, {y_encoded.sum()} positive cases")

# If label is not binary, skip modeling early
if y.nunique() != 2:
    print("! Non-binary label detected; Phase 4 models require a binary target. Skipping.")
    with open(OUTPUT_DIR / 'REPORT.md', 'w') as f:
        f.write('# PHASE 4: REGULARIZED MODELLING\n\nNon-binary label detected; modeling skipped.')
    pd.DataFrame().to_csv(OUTPUT_DIR / 'model_results.csv', index=False)
    print("  ✓ Wrote minimal outputs for skipped Phase 4")
    raise SystemExit(0)

# ============================================================================
# DEFINE MODELS
# ============================================================================

print("\n[2] Defining models...")

models_config = {
    'ElasticNet': {
        'model': lambda: Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=2000, random_state=42, class_weight='balanced',
                solver='saga', penalty='elasticnet'
            )),
        ]),
        'param_grid': {
            'clf__C': [0.01, 0.1, 1.0, 3.0, 10.0],
            'clf__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        'type': 'linear',
    },
    'LASSO': {
        'model': lambda: Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=2000, random_state=42, class_weight='balanced',
                solver='saga', penalty='l1'
            )),
        ]),
        'param_grid': {
            'clf__C': [0.01, 0.1, 1.0, 3.0, 10.0],
        },
        'type': 'linear',
    },
    'Ridge': {
        'model': lambda: Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=2000, random_state=42, class_weight='balanced',
                solver='lbfgs', penalty='l2'
            )),
        ]),
        'param_grid': {
            'clf__C': [0.01, 0.1, 1.0, 3.0, 10.0],
        },
        'type': 'linear',
    },
    'LinearSVM': {
        'model': lambda: Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel='linear', random_state=42, class_weight='balanced',
                probability=True
            )),
        ]),
        'param_grid': {
            'clf__C': [0.01, 0.1, 1.0, 3.0, 10.0],
        },
        'type': 'linear',
    },
    'RandomForest': {
        'model': lambda: RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42,
            class_weight='balanced', n_jobs=-1
        ),
        'param_grid': {
            'n_estimators': [200, 400, 800],
            'max_depth': [4, 6, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        'type': 'tree',
    },
}

print(f"✓ {len(models_config)} models configured")

# ============================================================================
# NESTED CV WITH STABILITY SELECTION
# ============================================================================

print("\n[3] Running nested CV with stability selection...")

cv_outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_folds = 5 * 3

model_results = {}
feature_importance_by_model = {name: {} for name in models_config.keys()}

for model_name, model_config in models_config.items():
    print(f"\n  Processing '{model_name}'...")
    
    fold_metrics = []
    selected_features_by_fold = []
    importances_by_fold = []
    best_params_by_fold = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_outer.split(X, y_encoded)):
        if (fold_idx + 1) % 5 == 0:
            print(f"    Fold {fold_idx+1}/{n_folds}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded.iloc[train_idx], y_encoded.iloc[test_idx]

        estimator = model_config['model']()
        param_grid = model_config['param_grid']
        try:
            search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring='roc_auc',
                cv=cv_inner,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best_params_by_fold.append(search.best_params_)
        except Exception as e:
            print(f"      ! Model fitting failed: {e}")
            continue
        
        # Predictions
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except:
            y_pred_proba = model.decision_function(X_test)
            if y_pred_proba.min() < 0:
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
        
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # PR-AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc_pr(recall, precision)
        
        fold_metrics.append({
            'fold': fold_idx,
            'roc_auc': auc_score,
            'pr_auc': pr_auc,
            'balanced_acc': bal_acc,
            'mcc': mcc,
            'best_params': json.dumps(search.best_params_, sort_keys=True),
            'best_cv_auc': search.best_score_,
        })
        
        # Feature importance/selection
        if model_name != 'RandomForest':
            # Linear models: use absolute coefficients from the final classifier
            coef_model = model.named_steps['clf'] if hasattr(model, 'named_steps') else model
            if hasattr(coef_model, 'coef_'):
                coef = np.abs(coef_model.coef_.flatten())
                importances_by_fold.append(dict(zip(selected_features, coef)))
        elif hasattr(model, 'feature_importances_'):
            # Tree models: use feature importance
            importance = model.feature_importances_
            importances_by_fold.append(dict(zip(selected_features, importance)))
        
        selected_features_by_fold.append(set(selected_features))
    
    # Aggregate results
    if fold_metrics:
        fold_metrics_df = pd.DataFrame(fold_metrics)
        model_results[model_name] = {
            'mean_auc': fold_metrics_df['roc_auc'].mean(),
            'std_auc': fold_metrics_df['roc_auc'].std(),
            'mean_pr_auc': fold_metrics_df['pr_auc'].mean(),
            'std_pr_auc': fold_metrics_df['pr_auc'].std(),
            'mean_bal_acc': fold_metrics_df['balanced_acc'].mean(),
            'std_bal_acc': fold_metrics_df['balanced_acc'].std(),
            'mean_mcc': fold_metrics_df['mcc'].mean(),
            'std_mcc': fold_metrics_df['mcc'].std(),
            'best_params': best_params_by_fold,
            'fold_metrics': fold_metrics,
        }
        
        # Aggregate feature importance
        if importances_by_fold:
            feature_importance_dict = {}
            for feature in numeric_cols:
                importances = [imp[feature] if feature in imp else 0 for imp in importances_by_fold]
                feature_importance_dict[feature] = {
                    'mean': np.mean(importances),
                    'std': np.std(importances),
                }
            feature_importance_by_model[model_name] = feature_importance_dict
        
        print(f"    ✓ ROC-AUC: {model_results[model_name]['mean_auc']:.3f}±{model_results[model_name]['std_auc']:.3f}")
        print(f"    ↳ Tuned params tried on {len(best_params_by_fold)} folds")

# If label is not binary, exit gracefully (nothing to model)
if y.nunique() != 2:
    print("! Non-binary label detected; Phase 4 models are binary-only. Skipping.")
    with open(OUTPUT_DIR / 'REPORT.md', 'w') as f:
        f.write('# PHASE 4: REGULARIZED MODELLING\n\nNon-binary label detected; modeling skipped.')
    pd.DataFrame().to_csv(OUTPUT_DIR / 'model_results.csv', index=False)
    print("  ✓ Wrote minimal outputs for skipped Phase 4")
    raise SystemExit(0)

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n[4] Creating visualizations...")

model_names = list(model_results.keys())

# 4a. ROC-AUC comparison across models
fig, ax = plt.subplots(figsize=(10, 6))
aucs = [model_results[m]['mean_auc'] for m in model_names]
auc_stds = [model_results[m]['std_auc'] for m in model_names]

colors = sns.color_palette("husl", len(model_names))
ax.bar(range(len(model_names)), aucs, yerr=auc_stds, capsize=5, alpha=0.7,
       color=colors, edgecolor='black', linewidth=1)
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.set_ylabel('ROC-AUC')
ax.set_ylim([0.4, 0.8])
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Random')
ax.set_title('ROC-AUC Across Models (Nested CV)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_model_comparison_auc.png', dpi=100, bbox_inches='tight')
print("  ✓ Saved 01_model_comparison_auc.png")
plt.close()

# 4b. PR-AUC comparison
fig, ax = plt.subplots(figsize=(10, 6))
pr_aucs = [model_results[m]['mean_pr_auc'] for m in model_names]
pr_auc_stds = [model_results[m]['std_pr_auc'] for m in model_names]

ax.bar(range(len(model_names)), pr_aucs, yerr=pr_auc_stds, capsize=5, alpha=0.7,
       color=colors, edgecolor='black', linewidth=1)
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.set_ylabel('PR-AUC')
ax.set_title('Precision-Recall AUC Across Models (Nested CV)')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_model_comparison_pr_auc.png', dpi=100, bbox_inches='tight')
print("  ✓ Saved 02_model_comparison_pr_auc.png")
plt.close()

# 4c. Feature importance for linear models (Elastic Net)
if feature_importance_by_model['ElasticNet']:
    top_n = 20
    elastic_imp = feature_importance_by_model['ElasticNet']
    elastic_imp_sorted = sorted(elastic_imp.items(), key=lambda x: -x[1]['mean'])[:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    features_plot = [f for f, _ in elastic_imp_sorted]
    importances_plot = [v['mean'] for _, v in elastic_imp_sorted]
    importances_std = [v['std'] for _, v in elastic_imp_sorted]
    
    ax.barh(range(len(features_plot)), importances_plot, xerr=importances_std, 
            capsize=3, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(features_plot)))
    ax.set_yticklabels(features_plot, fontsize=9)
    ax.set_xlabel('Mean |Coefficient|')
    ax.set_title(f'Top {top_n} Features - Elastic Net Logistic Regression')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_elasticnet_feature_importance.png', dpi=100, bbox_inches='tight')
    print("  ✓ Saved 03_elasticnet_feature_importance.png")
    plt.close()

# 4d. Feature importance for Random Forest
if feature_importance_by_model['RandomForest']:
    top_n = 20
    rf_imp = feature_importance_by_model['RandomForest']
    rf_imp_sorted = sorted(rf_imp.items(), key=lambda x: -x[1]['mean'])[:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    features_plot = [f for f, _ in rf_imp_sorted]
    importances_plot = [v['mean'] for _, v in rf_imp_sorted]
    importances_std = [v['std'] for _, v in rf_imp_sorted]
    
    ax.barh(range(len(features_plot)), importances_plot, xerr=importances_std,
            capsize=3, alpha=0.7, color='forestgreen', edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(features_plot)))
    ax.set_yticklabels(features_plot, fontsize=9)
    ax.set_xlabel('Mean Feature Importance')
    ax.set_title(f'Top {top_n} Features - Random Forest')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_randomforest_feature_importance.png', dpi=100, bbox_inches='tight')
    print("  ✓ Saved 04_randomforest_feature_importance.png")
    plt.close()

# 4e. Multi-metric comparison heatmap
metrics_matrix = []
for model_name in model_names:
    metrics_matrix.append([
        model_results[model_name]['mean_auc'],
        model_results[model_name]['mean_pr_auc'],
        model_results[model_name]['mean_bal_acc'],
        model_results[model_name]['mean_mcc'],
    ])

metrics_df = pd.DataFrame(metrics_matrix, index=model_names,
                         columns=['ROC-AUC', 'PR-AUC', 'Balanced Acc', 'MCC'])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
            cbar_kws={'label': 'Score'}, ax=ax, vmin=0.3, vmax=0.8)
ax.set_title('Model Performance Metrics (Nested CV)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_metrics_heatmap.png', dpi=100, bbox_inches='tight')
print("  ✓ Saved 05_metrics_heatmap.png")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n[5] Saving results...")

# Model comparison summary
summary_list = []
for model_name in model_names:
    summary_list.append({
        'Model': model_name,
        'Mean_AUC': model_results[model_name]['mean_auc'],
        'Std_AUC': model_results[model_name]['std_auc'],
        'Mean_PR_AUC': model_results[model_name]['mean_pr_auc'],
        'Mean_BalAcc': model_results[model_name]['mean_bal_acc'],
        'Mean_MCC': model_results[model_name]['mean_mcc'],
    })

summary_export = pd.DataFrame(summary_list)
summary_export.to_csv(OUTPUT_DIR / 'model_summary.csv', index=False)
print("  ✓ Saved model_summary.csv")

# Detailed fold results
all_fold_results = []
for model_name in model_names:
    for fold_result in model_results[model_name]['fold_metrics']:
        fold_result['model'] = model_name
        all_fold_results.append(fold_result)

fold_results_df = pd.DataFrame(all_fold_results)
fold_results_df.to_csv(OUTPUT_DIR / 'cv_fold_results.csv', index=False)
print("  ✓ Saved cv_fold_results.csv")

# Feature importance summary (top features across models)
all_importances = {}
for feature in numeric_cols:
    importances_list = []
    for model_name in model_names:
        if feature in feature_importance_by_model[model_name]:
            importances_list.append(feature_importance_by_model[model_name][feature]['mean'])
    if importances_list:
        all_importances[feature] = np.mean(importances_list)

top_features = sorted(all_importances.items(), key=lambda x: -x[1])[:30]
top_features_df = pd.DataFrame(top_features, columns=['Feature', 'Mean_Importance'])
top_features_df.to_csv(OUTPUT_DIR / 'top_features_by_importance.csv', index=False)
print("  ✓ Saved top_features_by_importance.csv")

write_feature_manifest(
    OUTPUT_DIR / 'feature_manifest_used.json',
    name='phase4_used_feature_set',
    source_phase='phase4_regularized',
    selection_rule=feature_source,
    features=selected_features,
    metadata={
        'feature_mode': args.feature_mode,
        'feature_source': feature_source,
        'n_features': len(selected_features),
        'model_count': len(models_config),
    },
)
print(f"  ✓ Saved feature_manifest_used.json ({len(selected_features)} features)")

# ============================================================================
# GENERATE REPORT
# ============================================================================

print("\n[6] Generating report...")

report = f"""
# PHASE 4: REGULARIZED MODELLING & STABILITY SELECTION

## Study Design
- **Cross-validation**: 3×5-fold repeated stratified CV ({n_folds} folds)
- **Models tested**: {len(model_names)}
- **Feature mode**: {args.feature_mode}
- **Feature source**: {feature_source}
- **Features**: {len(selected_features)}
- **Strategy**: Emphasis on Elastic Net (primary), with comparison to LASSO, Ridge, SVM, RF

## Model Performance Summary
"""

with open(OUTPUT_DIR / 'REPORT.md', 'w') as f:
    f.write(report)
print("  ✓ Saved REPORT.md")

with open(OUTPUT_DIR / 'summary.json', 'w') as f:
    json.dump({'n_models': len(model_names), 'n_folds': n_folds}, f, indent=2)
print("  ✓ Saved summary.json")

print("\n" + "="*70)
print("✓ PHASE 4 COMPLETE")
print("="*70)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
