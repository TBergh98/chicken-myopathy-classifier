
# PHASE 2: UNIVARIATE STATISTICS REPORT

## Overview
- **Features analyzed**: 164
- **Significant features (FDR<0.05, U-test)**: 15

## Top 20 Features by ROC-AUC

| Rank | Feature | ROC-AUC | Cohen's d | U-test p | Significant |
|------|---------|---------|-----------|----------|-------------|
| 1 | mfcc_18_min | 0.6608 | 0.5748 | 1.50e-04 | ✓ |
| 2 | mfcc_18_p10 | 0.6546 | 0.4842 | 2.68e-04 | ✓ |
| 3 | mfcc_17_p10 | 0.6541 | 0.5712 | 2.79e-04 | ✓ |
| 4 | mfcc_18_mean | 0.6491 | 0.5176 | 4.41e-04 | ✓ |
| 5 | mfcc_17_mean | 0.6466 | 0.5548 | 5.47e-04 | ✓ |
| 6 | mfcc_18_p90 | 0.6436 | 0.5200 | 7.09e-04 | ✓ |
| 7 | mfcc_17_median | 0.6399 | 0.5390 | 9.76e-04 | ✓ |
| 8 | mfcc_18_median | 0.6395 | 0.4953 | 1.01e-03 | ✓ |
| 9 | mfcc_19_median | 0.6386 | 0.4374 | 1.09e-03 | ✓ |
| 10 | mfcc_19_mean | 0.6371 | 0.4128 | 1.23e-03 | ✓ |
| 11 | mfcc_17_p90 | 0.6317 | 0.4886 | 1.90e-03 | ✓ |
| 12 | mfcc_19_p10 | 0.6257 | 0.3710 | 3.04e-03 | ✓ |
| 13 | mfcc_15_max | 0.6255 | 0.5012 | 3.09e-03 | ✓ |
| 14 | mfcc_11_min | 0.6189 | 0.3725 | 5.08e-03 | - |
| 15 | mfcc_19_p90 | 0.6159 | 0.3763 | 6.27e-03 | - |
| 16 | mfcc_17_min | 0.6157 | 0.4220 | 6.40e-03 | - |
| 17 | mean_event_duration_sec | 0.6088 | 0.3772 | 1.03e-02 | - |
| 18 | mfcc_19_min | 0.6067 | 0.4052 | 1.19e-02 | - |
| 19 | mfcc_10_min | 0.6066 | 0.4104 | 1.20e-02 | - |
| 20 | mfcc_15_mean | 0.6042 | 0.3578 | 1.40e-02 | - |

## Top 20 Features by Effect Size (Cohen's d)

| Rank | Feature | Cohen's d | ROC-AUC | Median Diff |
|------|---------|-----------|---------|------------|
| 1 | mfcc_18_min | 0.5748 | 0.6608 | 2.8731 |
| 2 | mfcc_17_p10 | 0.5712 | 0.6541 | 1.6477 |
| 3 | mfcc_17_mean | 0.5548 | 0.6466 | 0.8314 |
| 4 | mfcc_17_median | 0.5390 | 0.6399 | 0.9194 |
| 5 | mfcc_18_p90 | 0.5200 | 0.6436 | 1.9190 |
| 6 | mfcc_18_mean | 0.5176 | 0.6491 | 1.9118 |
| 7 | mfcc_15_max | 0.5012 | 0.6255 | 2.7061 |
| 8 | mfcc_18_median | 0.4953 | 0.6395 | 1.7431 |
| 9 | mfcc_17_p90 | 0.4886 | 0.6317 | 0.7590 |
| 10 | mfcc_18_p10 | 0.4842 | 0.6546 | 1.8923 |
| 11 | mfcc_19_median | 0.4374 | 0.6386 | 1.0576 |
| 12 | mfcc_01_std | -0.4302 | 0.3669 | -4.5895 |
| 13 | mfcc_05_median | -0.4277 | 0.3944 | -2.9953 |
| 14 | mfcc_17_min | 0.4220 | 0.6157 | 2.4025 |
| 15 | mfcc_19_mean | 0.4128 | 0.6371 | 0.9754 |
| 16 | mfcc_10_min | 0.4104 | 0.6066 | 2.2173 |
| 17 | zcr_std | -0.4070 | 0.3764 | -0.0025 |
| 18 | mfcc_19_min | 0.4052 | 0.6067 | 3.1620 |
| 19 | vocalization_rate_per_min | -0.3943 | 0.3932 | -11.4235 |
| 20 | mfcc_16_mean | 0.3933 | 0.5974 | 0.4739 |

## Statistical Summary
- **Features with AUC > 0.55**: 52
- **Features with AUC > 0.60**: 21
- **Features with |Cohen's d| > 0.5**: 7
- **Mean ROC-AUC**: 0.5068
- **Median ROC-AUC**: 0.5011

## Visualizations
- `01_volcano_plot.png` - Effect size vs statistical significance
- `02_roc_auc_ranking.png` - Single-feature prediction performance
- `03_cohend_ranking.png` - Feature effect sizes
- `04_pvalue_auc_distributions.png` - P-value and AUC distributions

## Key Findings
- Univariate analysis identifies features that independently differ between classes
- ROC-AUC > 0.55 indicates modest predictive power
- Significant features warrant inclusion in multivariate models
- Non-significant features may still contribute to multivariate predictions

## Next Steps
- Phase 3: Feature block analysis to compare feature group predictive value
- Phase 4: Regularized modeling with stability selection
