
# PHASE 3: FEATURE BLOCK ANALYSIS REPORT

## Study Design
- **Cross-validation**: 3×5-fold repeated stratified CV
- **Total folds**: 15
- **Model**: Logistic Regression with elastic net regularization
- **Evaluation metrics**: ROC-AUC, Balanced Accuracy, MCC

## Results Summary

| Feature Group | N Features | Mean ROC-AUC | Mean Bal. Acc | Mean MCC |
|---|---:|---:|---:|---:|
| Temporal | 6 | 0.5733±0.0887 | 0.5445±0.0900 | 0.0803±0.1639 |
| Energy | 4 | 0.5381±0.0602 | 0.5216±0.0658 | 0.0397±0.1205 |
| Spectral | 10 | 0.6224±0.0932 | 0.6141±0.0818 | 0.2080±0.1490 |
| F0 | 5 | 0.5931±0.0897 | 0.5785±0.0698 | 0.1435±0.1268 |
| MFCC | 140 | 0.5754±0.0956 | 0.5463±0.0676 | 0.0912±0.1366 |
| Temporal+Spectral | 16 | 0.6332±0.0943 | 0.6221±0.0998 | 0.2240±0.1826 |
| Temporal+Spectral+F0 | 21 | 0.6367±0.0951 | 0.6170±0.0853 | 0.2155±0.1573 |
| All Features | 164 | 0.6095±0.0902 | 0.5358±0.0859 | 0.0692±0.1715 |
| PCA-reduced | 10 | 0.6115±0.0891 | 0.6180±0.0658 | 0.2173±0.1217 |

## Key Findings

### Best Performing Feature Group
- **Temporal+Spectral+F0** with ROC-AUC = 0.6367±0.0951

## Visualizations
- `01_roc_auc_by_group.png` - ROC-AUC comparison
- `02_balanced_accuracy_by_group.png` - Balanced accuracy comparison
- `03_mcc_by_group.png` - MCC comparison  
- `04_metrics_heatmap.png` - Multi-metric heatmap

## Next Steps
- Phase 4: Regularized modeling with all approaches + stability selection
- Phase 5: Feature importance interpretation
