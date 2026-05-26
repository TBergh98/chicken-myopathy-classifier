# Feature Selection & Dimensionality Reduction - Implementation Guide

## Overview

This document describes the **6-phase feature selection and dimensionality reduction pipeline** for acoustic biomarker discovery in poultry muscular myopathy. The pipeline directly implements the recommendations from `docs/feature_selection.md` while prioritizing:

1. **Precision over strict interpretability** (Elastic Net as primary model)
2. **Careful treatment of energy features** (flagged, not discarded)
3. **Conservative outlier detection** (Z-score, not Isolation Forest)
4. **Block-wise modeling** for biological interpretability
5. **Repeated nested cross-validation** to prevent overfitting
6. **Organized, understandable outputs**

---

## Quick Start

### Prerequisites

Ensure you have the conda environment activated:

```bash
conda activate vocalizzazioni-audio
```

Verify required packages:

```bash
pip install scikit-learn pandas numpy scipy matplotlib seaborn optuna shap
```

### Run Full Pipeline

Execute all phases (1-4) in sequence:

```bash
python run_analysis.py
```

Or run individual phases:

```bash
python 01_phase1_eda.py
python 02_phase2_univariate.py
python 03_phase3_blocks.py
python 04_phase4_regularized.py
```

---

## Pipeline Structure

```
analysis/
├── phase1_eda/                    # Exploratory Data Analysis
│   ├── REPORT.md
│   ├── metadata.json
│   ├── 01_class_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_top_features_by_class.png
│   ├── 04_feature_blocks_by_class.png
│   └── 05_feature_variance.png
│
├── phase2_univariate/              # Univariate Statistics
│   ├── REPORT.md
│   ├── summary.json
│   ├── univariate_results.csv      # Full results table
│   ├── significant_features.csv    # FDR < 0.05
│   ├── 01_volcano_plot.png
│   ├── 02_roc_auc_ranking.png
│   ├── 03_cohend_ranking.png
│   └── 04_pvalue_auc_distributions.png
│
├── phase3_blocks/                  # Feature Block Analysis
│   ├── REPORT.md
│   ├── summary.json
│   ├── group_summary.csv           # Performance by feature group
│   ├── cv_fold_results.csv
│   ├── 01_roc_auc_by_group.png
│   ├── 02_balanced_accuracy_by_group.png
│   ├── 03_mcc_by_group.png
│   └── 04_metrics_heatmap.png
│
├── phase4_regularized/             # Regularized Modelling
│   ├── REPORT.md
│   ├── summary.json
│   ├── model_summary.csv           # Model comparison
│   ├── cv_fold_results.csv
│   ├── top_features_by_importance.csv
│   ├── 01_model_comparison_auc.png
│   ├── 02_model_comparison_pr_auc.png
│   ├── 03_elasticnet_feature_importance.png
│   ├── 04_randomforest_feature_importance.png
│   └── 05_metrics_heatmap.png
│
├── phase5_interpret/               # [Planned] Interpretation
├── phase6_sensitivity/             # [Planned] Sensitivity Analysis
└── ANALYSIS_SUMMARY.md             # Master summary

analysis_scripts/                   # Supporting scripts
├── 00_setup.py
└── utils.py                        # [Future] Shared utilities
```
