
# PHASE 1: EXPLORATORY DATA ANALYSIS REPORT

## Dataset Overview
- **Total samples**: 229
- **Total features**: 159
- **Feature blocks**: 5

## Feature Blocks

### Temporal (6 features)
duration_sec, sample_rate, silence_percent, num_high_energy_events, vocalization_rate_per_min, ... (+1 more)

### Energy (4 features)
rms_mean, rms_std, rms_max, num_high_energy_events

### Spectral (10 features)
zcr_mean, zcr_std, spectral_centroid_mean, spectral_centroid_std, spectral_bandwidth_mean, ... (+5 more)

### F0 (5 features)
f0_mean, f0_std, f0_min, f0_max, f0_valid_ratio

### MFCC (140 features)
mfcc_01_mean, mfcc_01_std, mfcc_01_min, mfcc_01_max, mfcc_01_median, ... (+135 more)

## Target Variable: ws_myopathy_binary
- **Classes**: 2
- **Distribution**: 
  - 0: 164 (71.6%)
  - 1: 65 (28.4%)


## Data Quality
- **Missing values**: 0
- **Duplicate rows**: 0
- **Outliers (Z > 3)**: 116 features with outliers

## High Correlation Pairs (|r| > 0.85)
- **Count**: 94

Top 10 correlated pairs:
- vocalization_rate_per_min <-> mean_event_duration_sec: -0.999
- mfcc_08_mean <-> mfcc_08_median: 0.996
- mfcc_18_mean <-> mfcc_18_median: 0.995
- mfcc_15_mean <-> mfcc_15_median: 0.995
- mfcc_07_mean <-> mfcc_07_median: 0.994
- mfcc_10_mean <-> mfcc_10_median: 0.993
- mfcc_13_mean <-> mfcc_13_median: 0.992
- mfcc_17_mean <-> mfcc_17_median: 0.992
- mfcc_16_mean <-> mfcc_16_median: 0.992
- mfcc_14_mean <-> mfcc_14_median: 0.992


## Visualizations Created
1. `01_class_distribution.png` - Class balance and counts
2. `02_correlation_heatmap.png` - Correlation structure
3. `03_top_features_by_class.png` - Feature distributions by class
4. `04_feature_blocks_by_class.png` - Block-level comparison
5. `05_feature_variance.png` - Top 20 features by variance

## Next Steps
- Phase 2: Univariate statistical tests
- Phase 3: Feature block analysis
- Phase 4: Regularized modeling with feature selection