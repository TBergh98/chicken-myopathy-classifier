# Session e1a3d656-1424-4345-acf3-8a4c1fb13e85

- Model: openai/gpt-5.5

---

### Assistant

Below is a practical overview tailored to your dataset: **250 hens × ~170 acoustic summary features**, with a binary outcome: **myopathy present/absent**.

Key point first: I found **no strong direct literature showing that hen/chicken vocalizations are already validated biomarkers of muscular myopathies** such as woody breast, white striping, spaghetti meat, etc. Existing poultry/bioacoustics literature is much stronger for **welfare, stress, thermal comfort, growth/body weight, distress calls, respiratory monitoring, and general animal-sound classification**. So your problem should be framed as an **exploratory biomarker/predictive modelling study**, not as a replication of an established myopathy-vocalization marker.

---

# 1. Your dataset structure and main statistical issue

You have approximately:

- **n = 250 hens**
- **p ≈ 170 acoustic features**
- Binary label: myopathy presence/absence

This is a **small-n, moderate-p** tabular problem. The danger is not that you lack features; the danger is that you have **too many correlated acoustic summaries** relative to the number of hens.

Most features are likely highly correlated:

- `mfcc_01_mean`, `mfcc_01_median`, `mfcc_01_p10`, `mfcc_01_p90`, etc.
- Spectral centroid / bandwidth / rolloff are often correlated.
- RMS features correlate with recording distance/gain.
- F0 features may be unreliable when calls are noisy or unvoiced.

So the central goal should be:

> Identify a small, stable, interpretable set of acoustic variables or low-dimensional components that separate myopathic vs non-myopathic hens, while avoiding overfitting.

---

# 2. Pre-analysis checks before feature selection

Before any feature selection, I strongly recommend these checks.

## 2.1 Remove non-informative identifiers

Do **not** use:

```text
chicken_id
```

as a model feature.

It is an identifier, not a biological predictor. Keep it only for traceability.

## 2.2 Check label distribution

You need to know:

```text
myopathy present: n = ?
myopathy absent: n = ?
```

If the classes are imbalanced, e.g. 40 diseased vs 210 healthy, accuracy becomes misleading. Prefer:

- ROC-AUC
- PR-AUC
- balanced accuracy
- sensitivity/specificity
- F1
- MCC
- calibrated probability curves

## 2.3 Inspect missing values

Especially important for:

```text
f0_mean
f0_std
f0_min
f0_max
f0_valid_ratio
```

F0 extraction can fail if the signal is noisy, broadband, or unvoiced. Missing or unreliable F0 values may themselves be informative, but you should distinguish:

- true biological absence of tonal vocalization
- extraction failure
- low-quality recording

## 2.4 Standardize all numeric features

Most methods require scaling:

```text
z = (x - mean) / std
```

This is essential for:

- PCA
- logistic regression
- SVM
- LASSO / elastic net
- kNN
- PLS-DA
- t-SNE / UMAP

Tree-based models technically do not require it, but for a unified workflow it is still useful.

## 2.5 Detect outliers

Audio features are often dominated by recording artefacts.

Check:

- extreme `duration_sec`
- extreme `rms_max`
- very high `silence_percent`
- zero or near-zero `vocalization_rate_per_min`
- suspicious `sample_rate` differences
- extreme spectral centroid/rolloff
- impossible F0 values

Use:

- boxplots
- robust z-scores
- PCA outlier plots
- Mahalanobis distance
- isolation forest only for diagnostic purposes

---

# 3. Feature families in your dataset

Your features can be grouped into biologically meaningful blocks.

## 3.1 Recording-level/control features

```text
duration_sec
sample_rate
```

These should usually not be interpreted biologically.

- `duration_sec`: may affect number of detected events.
- `sample_rate`: should ideally be constant. If not, it can introduce artefacts.

Use them for quality control, not necessarily as predictors.

---

## 3.2 Energy/intensity features

```text
rms_mean
rms_std
rms_max
num_high_energy_events
```

Potential interpretation:

- louder or more forceful vocalizations
- agitation/distress
- microphone distance/gain artefact
- movement or environmental noise

Caution: RMS is highly sensitive to recording setup. If microphone distance and gain were not standardized, energy features may reflect technical variation.

---

## 3.3 Temporal/vocal activity features

```text
silence_percent
num_high_energy_events
vocalization_rate_per_min
mean_event_duration_sec
duration_sec
```

These are among the most promising for your goal because they summarize behavioural activity.

Potential interpretation:

- sick or impaired hens may vocalize less/more
- isolation distress may differ by health state
- myopathic birds may show altered activity or stress responses

Important features:

```text
silence_percent
vocalization_rate_per_min
mean_event_duration_sec
num_high_energy_events
```

These should be among your first candidates.

---

## 3.4 Spectral-shape features

```text
zcr_mean
zcr_std
spectral_centroid_mean
spectral_centroid_std
spectral_bandwidth_mean
spectral_bandwidth_std
spectral_rolloff_mean
spectral_rolloff_std
spectral_flatness_mean
spectral_flatness_std
```

Potential interpretation:

- higher centroid/rolloff: more high-frequency energy
- higher bandwidth: broader/noisier calls
- higher flatness: more noise-like signal
- ZCR: rough high-frequency/noisiness proxy

These could reflect:

- harshness of calls
- respiratory/noisy sounds
- stress-related vocal changes
- recording noise

Promising features:

```text
spectral_centroid_mean
spectral_rolloff_mean
spectral_bandwidth_mean
spectral_flatness_mean
zcr_mean
```

Their standard deviations may also be useful because variability across the recording can reflect irregular calling patterns.

---

## 3.5 MFCC features

You have for each MFCC 1–20:

```text
mean
std
min
max
median
p10
p90
```

This is the largest block of features.

MFCCs summarize spectral envelope/timbre. They are widely used in:

- speech analysis
- animal vocalization classification
- bird/bioacoustic classification
- sound-event recognition

But in your case they create a high-dimensional correlated block:

```text
20 MFCCs × 7 statistics = 140 features
```

This block alone is more than half your dataset size. You should **not** interpret all MFCC features individually without reduction or stability analysis.

Recommended handling:

1. Use only MFCC means and standard deviations initially.
2. Or apply PCA only to the MFCC block.
3. Or use elastic-net/LASSO stability selection.
4. Or group them as a feature block and compare model performance with/without MFCCs.

I would avoid starting with all 140 MFCC-derived variables in an unregularized model.

---

## 3.6 Fundamental frequency/pitch features

```text
f0_mean
f0_std
f0_min
f0_max
f0_valid_ratio
```

Potential interpretation:

- pitch shifts
- tonal vs noisy calls
- physiological/vocal tract differences
- stress-related changes
- call-type composition

Most important:

```text
f0_valid_ratio
f0_mean
f0_std
f0_max - f0_min
```

I would create an additional derived feature:

```text
f0_range = f0_max - f0_min
```

Caution: Chicken vocalizations can be noisy, and F0 estimation may be unstable. Always examine `f0_valid_ratio`.

---

# 4. Dimensionality reduction strategies

Dimensionality reduction methods transform many features into fewer variables. Some are useful for interpretation; others only for visualization.

---

## 4.1 PCA: Principal Component Analysis

### What it does

PCA finds linear combinations of features that explain maximum variance.

### Useful for your dataset?

Yes. PCA is a good first method because your features are highly correlated.

### How to use it

Use PCA after standardization:

```text
StandardScaler → PCA
```

You can apply PCA to:

1. all numeric acoustic features
2. only the MFCC block
3. only spectral features
4. only temporal/energy features

### Recommended use

For your dataset, I would use **block-wise PCA**:

- PCA on MFCC features
- PCA on spectral features
- PCA on energy/event features
- PCA on F0 features if enough valid F0 data exist

Then use the first few PCs from each block in a classifier.

Example:

```text
MFCC_PC1, MFCC_PC2, MFCC_PC3
Spectral_PC1, Spectral_PC2
Energy_PC1
F0_PC1
```

This reduces ~170 variables to perhaps 8–15 components.

### Pros

- Handles correlated features.
- Reduces overfitting.
- Gives loadings showing which variables drive each component.
- Works well with small datasets.

### Cons

- Unsupervised: components maximize variance, not disease separation.
- PCs can be harder to interpret biologically.

### Best use

Use PCA for:

- exploratory analysis
- reducing MFCC block
- detecting outliers
- creating compact predictors

---

## 4.2 Sparse PCA

Sparse PCA is like PCA but forces many feature loadings to zero.

### Useful for your case?

Potentially yes, because it creates components that depend on fewer features.

### Pros

- More interpretable than standard PCA.
- Useful when features are highly redundant.

### Cons

- Requires tuning sparsity.
- Can be unstable with n = 250.
- Less standard than PCA.

Use it after simpler PCA and regularized models.

---

## 4.3 PLS-DA: Partial Least Squares Discriminant Analysis

PLS-DA is supervised dimensionality reduction. It finds components that maximize covariance between features and class labels.

### Useful for your case?

Yes, but with caution.

PLS-DA is common in omics/biomarker-style datasets where p is high and n is moderate. Your dataset is similar in structure.

### Pros

- Supervised: components are optimized for myopathy separation.
- Handles correlated predictors.
- Can rank feature importance through VIP scores.

### Cons

- Easy to overfit.
- Must be tuned inside cross-validation.
- PLS-DA plots can look overly optimistic if fitted on the full dataset.

### Recommendation

Use PLS-DA only with:

- repeated cross-validation
- nested tuning of number of components
- permutation testing

Do **not** fit PLS-DA once on the whole dataset and claim separation from the resulting plot.

---

## 4.4 LDA: Linear Discriminant Analysis

LDA finds a linear axis separating two groups.

### Useful?

For visualization, yes. For prediction, maybe.

With binary labels, LDA gives one discriminant axis.

### Pros

- Simple and interpretable.
- Good for visualizing class separation.

### Cons

- Assumes approximately Gaussian classes with similar covariance.
- With many correlated features, can be unstable.
- Needs dimensionality reduction first if p is high.

### Best workflow

```text
StandardScaler → PCA → LDA
```

Example:

```text
Use first 5–20 PCs, then LDA.
```

---

## 4.5 UMAP

UMAP is a nonlinear visualization method.

### Useful?

Yes, for exploratory visualization.

### Pros

- Can reveal nonlinear clustering.
- Often better than t-SNE for preserving some global structure.
- Good for visualizing whether myopathic/non-myopathic hens form clusters.

### Cons

- Not a statistical test.
- Sensitive to parameters.
- Can show apparent clusters even in weak data.
- Should not be used as proof of separability.

### Recommended use

Use:

```text
StandardScaler → PCA to 10–30 dimensions → UMAP to 2D
```

Color points by:

- myopathy present/absent
- severity score if available
- batch/farm/day
- body weight
- age
- recording duration
- microphone/session

This helps detect confounding.

---

## 4.6 t-SNE

t-SNE is another nonlinear visualization method.

### Useful?

Only for visualization.

### Pros

- Good at showing local neighborhoods.
- Can reveal subgroups.

### Cons

- Very sensitive to perplexity and random seed.
- Poor global distance interpretation.
- Not suitable for feature selection.
- Can produce misleading clusters.

### Recommendation

Use t-SNE only after PCA and only as a supplementary plot.

---

## 4.7 Autoencoders

With only 250 hens, I would not recommend training a neural autoencoder on these tabular features.

### Why?

- Too little data.
- High overfitting risk.
- Hard to interpret.
- PCA will likely be more stable.

Autoencoders may become useful only if you return to the raw audio and have many segmented calls.

---

# 5. Feature selection strategies

Feature selection keeps original features rather than transforming them.

This is important because you likely want interpretable acoustic biomarkers.

---

## 5.1 Filter methods: simple univariate screening

These methods test each feature individually against the label.

### Options

For each feature, compare myopathic vs non-myopathic hens using:

- t-test, if approximately normal
- Mann–Whitney U test, if non-normal
- point-biserial correlation
- ROC-AUC per feature
- mutual information
- Cohen’s d or Cliff’s delta effect size

### Recommended outputs

For each feature:

```text
feature
mean/median in healthy
mean/median in myopathic
effect size
p-value
FDR-adjusted p-value
single-feature ROC-AUC
```

### Pros

- Very interpretable.
- Good first screening.
- Helps identify candidate biomarkers.

### Cons

- Ignores interactions between features.
- Correlated features create redundancy.
- Multiple testing problem.

### Important

You have ~170 features. Apply correction:

```text
Benjamini-Hochberg FDR
```

not only raw p-values.

### Features to prioritize in univariate analysis

Start with:

```text
silence_percent
vocalization_rate_per_min
num_high_energy_events
mean_event_duration_sec
rms_mean
rms_std
spectral_centroid_mean
spectral_bandwidth_mean
spectral_rolloff_mean
spectral_flatness_mean
zcr_mean
f0_mean
f0_std
f0_valid_ratio
```

Then examine MFCC features.

---

## 5.2 Correlation-based pruning

Before model-based selection, remove redundant features.

### Method

Compute Spearman or Pearson correlation between all features.

If two features have:

```text
|r| > 0.85 or |r| > 0.90
```

keep only one.

For example, these may be redundant:

```text
mfcc_01_mean
mfcc_01_median
mfcc_01_p10
mfcc_01_p90
```

### How to choose which one to keep

Prefer features that are:

1. more interpretable
2. less missing
3. more stable
4. stronger univariate association with label
5. biologically plausible

### Pros

- Reduces redundancy.
- Makes models more stable.
- Helps interpretation.

### Cons

- Does not consider multivariate interactions.

---

## 5.3 Variance thresholding

Remove features with near-zero variance.

Examples:

- constant `sample_rate`
- features with almost identical values across hens
- F0 features if most calls failed

This should be done before more advanced methods.

---

## 5.4 LASSO logistic regression

LASSO logistic regression uses L1 regularization to shrink some coefficients to zero.

### Useful?

Very useful for your dataset.

### Model

```text
myopathy ~ selected acoustic features
```

with L1 penalty.

### Pros

- Performs classification and feature selection simultaneously.
- Good for high-dimensional correlated data.
- Produces a small feature set.

### Cons

- When features are highly correlated, LASSO may arbitrarily choose one of several equivalent features.
- Selected features can vary across folds.

### Recommendation

Use **stability selection**:

1. Run LASSO inside repeated cross-validation.
2. Record which features are selected.
3. Keep features selected in, e.g., >60–80% of folds.

This gives more reliable candidates.

---

## 5.5 Elastic-net logistic regression

Elastic net combines L1 and L2 penalties.

### Useful?

Probably one of the best first models for your dataset.

### Why?

Your features are correlated. Elastic net is more stable than pure LASSO.

### Tune

```text
C or lambda
l1_ratio
```

Example l1 ratios:

```text
0.1, 0.3, 0.5, 0.7, 0.9
```

### Pros

- Handles correlated predictors better than LASSO.
- Gives interpretable coefficients.
- Works well with small tabular datasets.
- Can include class weights for imbalance.

### Cons

- Still linear.
- Needs careful cross-validation.

---

## 5.6 Recursive Feature Elimination, RFE

RFE repeatedly fits a model and removes weak features.

### Common base models

- logistic regression
- linear SVM
- random forest
- gradient boosting

### Useful?

Yes, but computationally heavier and potentially unstable.

### Recommendation

Use RFE only after reducing obvious redundancy.

Example:

```text
correlation pruning → RFE with logistic regression/SVM → final model
```

### Pros

- Multivariate.
- Can find compact feature sets.

### Cons

- Can overfit if not nested inside CV.
- Feature rankings can be unstable with correlated variables.

---

## 5.7 Sequential feature selection

Forward selection:

1. Start with zero features.
2. Add the feature that improves CV score most.
3. Continue until performance stops improving.

Backward selection:

1. Start with all features.
2. Remove weakest features.

### Useful?

Maybe, but with caution.

For n = 250 and ~170 features, forward selection may be feasible after correlation pruning.

### Pros

- Directly optimizes model performance.
- Can produce small feature sets.

### Cons

- Greedy.
- Can overfit.
- Computationally expensive.
- Must be nested inside cross-validation.

---

## 5.8 Tree-based feature importance

Models:

- random forest
- extremely randomized trees
- gradient boosting
- XGBoost / LightGBM / CatBoost

### Useful?

Yes, especially for nonlinear relationships.

### Feature importance types

Avoid relying only on default Gini importance. Prefer:

```text
permutation importance
SHAP values
```

### Pros

- Captures nonlinear effects.
- Captures interactions.
- Handles mixed feature scales.

### Cons

- Can overfit with n = 250.
- Importance is unstable with correlated predictors.
- Default importance can be biased toward continuous/noisy features.

### Recommendation

Use tree models as a complementary approach, not the only method.

A good workflow:

```text
Random forest / gradient boosting → repeated CV → permutation importance on held-out folds
```

---

## 5.9 Permutation importance

Permutation importance measures how much model performance drops when one feature is randomly shuffled.

### Useful?

Very useful.

### Pros

- Model-agnostic.
- Evaluates importance based on predictive contribution.
- Can be computed on held-out folds.

### Cons

- Correlated features can hide each other’s importance.
- If two features contain the same information, permuting one may not reduce performance much.

### Recommendation

Use **grouped permutation importance**.

For example, permute whole feature groups:

```text
Energy block
Temporal/event block
Spectral block
MFCC block
F0 block
```

This may be more informative than single-feature importance.

---

## 5.10 SHAP analysis

SHAP can explain predictions from tree models or linear models.

### Useful?

Yes, after you have a reasonably validated model.

### Pros

- Shows directionality.
- Can reveal nonlinear thresholds.
- Useful for interpretation.

Example interpretation:

```text
High silence_percent increases predicted probability of myopathy.
Low f0_valid_ratio increases predicted probability of myopathy.
High spectral_flatness_mean increases predicted probability of myopathy.
```

### Cons

- Can be unstable in small datasets.
- Correlated features complicate interpretation.
- Should not replace statistical validation.

---

## 5.11 Stability selection

This is one of the most important strategies for your case.

### Idea

A feature is credible if it is repeatedly selected across many resampled datasets.

### Procedure

1. Repeat cross-validation many times.
2. Fit feature selection only on training folds.
3. Record selected features.
4. Rank by selection frequency.

Example output:

| Feature | Selection frequency |
|---|---:|
| vocalization_rate_per_min | 0.86 |
| spectral_flatness_mean | 0.78 |
| f0_valid_ratio | 0.74 |
| mfcc_03_std | 0.67 |
| silence_percent | 0.64 |

### Pros

- Reduces false discoveries.
- Handles small sample size better.
- Produces more trustworthy feature rankings.

### Cons

- Requires repeated modelling.
- Results depend on model family.

### Recommendation

Use stability selection with:

- elastic-net logistic regression
- linear SVM
- random forest/permutation importance

Then look for agreement across methods.

---

# 6. Visualization techniques

Visualization is essential, but it should not be confused with proof.

---

## 6.1 Feature distribution plots

For each candidate feature, plot:

- histogram/density by class
- boxplot
- violin plot
- swarm/strip plot

Best for:

```text
vocalization_rate_per_min
silence_percent
mean_event_duration_sec
rms_mean
spectral_centroid_mean
spectral_flatness_mean
f0_valid_ratio
f0_mean
```

These plots help identify:

- group differences
- outliers
- skewness
- threshold-like effects

---

## 6.2 Correlation heatmap

Make a correlation heatmap of all features or feature blocks.

Recommended:

- Spearman correlation
- hierarchical clustering of features

This will reveal redundancy among:

- MFCC statistics
- spectral features
- energy features

Useful for deciding which features to prune.

---

## 6.3 PCA score plot

Plot PC1 vs PC2, colored by:

- myopathy status
- batch
- recording session
- breed/line
- age
- body weight
- lesion score if available

Use this to answer:

- Are myopathic hens globally separated?
- Are there recording artefacts?
- Are there batch effects?
- Are outliers driving the result?

---

## 6.4 PCA loading plot

A PCA loading plot tells you which features contribute most to each principal component.

Example:

- PC1 may represent general vocal activity/energy.
- PC2 may represent spectral brightness/noisiness.
- PC3 may represent MFCC/timbre variation.

This helps convert abstract components into biological interpretations.

---

## 6.5 Volcano plot

For univariate analysis:

x-axis:

```text
effect size
```

y-axis:

```text
-log10(FDR-adjusted p-value)
```

This identifies features that are both statistically associated and biologically meaningful.

---

## 6.6 ROC curves for top single features

For each top feature, compute single-feature ROC-AUC.

Example:

```text
silence_percent alone: AUC = ?
vocalization_rate_per_min alone: AUC = ?
f0_valid_ratio alone: AUC = ?
spectral_flatness_mean alone: AUC = ?
```

This helps assess whether any one feature is informative.

---

## 6.7 UMAP/t-SNE plots

Use only as exploratory visualization.

Recommended pipeline:

```text
StandardScaler → PCA to 10–30 PCs → UMAP/t-SNE to 2D
```

Color by:

- myopathy status
- severity
- batch
- body weight
- age
- recording quality

If clusters align with batch rather than disease, that signals confounding.

---

## 6.8 SHAP summary plot

After fitting a validated tree-based model, use SHAP summary plots to show:

- most influential features
- whether high/low values increase risk
- nonlinear patterns

This can be very useful for communicating results.

---

# 7. Recommended modelling/selection workflow

For your dataset, I would use this pipeline.

---

## Step 1: Quality control

Remove:

```text
chicken_id
```

Check:

```text
duration_sec
sample_rate
missing values
outliers
f0_valid_ratio
class balance
```

Create derived features:

```text
f0_range = f0_max - f0_min
rms_range = rms_max - rms_mean
```

Possibly also:

```text
mfcc_i_range = mfcc_i_p90 - mfcc_i_p10
```

for each MFCC, but only if you later reduce dimensionality.

---

## Step 2: Define feature blocks

Use these blocks:

### Temporal/event block

```text
silence_percent
num_high_energy_events
vocalization_rate_per_min
mean_event_duration_sec
duration_sec
```

### Energy block

```text
rms_mean
rms_std
rms_max
```

### Spectral block

```text
zcr_mean
zcr_std
spectral_centroid_mean
spectral_centroid_std
spectral_bandwidth_mean
spectral_bandwidth_std
spectral_rolloff_mean
spectral_rolloff_std
spectral_flatness_mean
spectral_flatness_std
```

### F0 block

```text
f0_mean
f0_std
f0_min
f0_max
f0_valid_ratio
f0_range
```

### MFCC block

Either all MFCC features or a reduced subset.

I would initially use:

```text
mfcc_01_mean ... mfcc_20_mean
mfcc_01_std ... mfcc_20_std
```

Then add min/max/median/p10/p90 only if justified.

---

## Step 3: Univariate screening

For each feature:

- compute group medians
- compute effect size
- compute p-value
- apply FDR correction
- compute single-feature ROC-AUC

This gives interpretable candidate biomarkers.

---

## Step 4: Correlation pruning

Within highly correlated feature clusters, keep only one representative.

Threshold:

```text
|Spearman r| > 0.85
```

or stricter:

```text
|Spearman r| > 0.90
```

---

## Step 5: Fit regularized models

Use repeated stratified cross-validation.

Models:

1. elastic-net logistic regression
2. LASSO logistic regression
3. linear SVM
4. random forest
5. gradient boosting

Metrics:

- ROC-AUC
- PR-AUC
- balanced accuracy
- sensitivity
- specificity
- MCC

---

## Step 6: Stability selection

For each repeated CV fold, record selected/important features.

Rank features by:

```text
selection frequency
mean absolute coefficient
permutation importance
SHAP importance
```

Features repeatedly selected across methods are your strongest candidates.

---

## Step 7: Block-level modelling

Compare models using different feature groups:

| Model | Features |
|---|---|
| A | Temporal/event only |
| B | Energy only |
| C | Spectral only |
| D | F0 only |
| E | MFCC only |
| F | Temporal + spectral |
| G | Temporal + spectral + F0 |
| H | All features |
| I | PCA-reduced all features |

This tells you whether myopathy information is mostly in:

- vocal activity
- spectral quality
- pitch
- MFCC timbre
- energy

This is more biologically interpretable than throwing all features into one model.

---

# 8. Recommended validation strategy

Because n = 250, validation is critical.

## 8.1 Use nested cross-validation

Feature selection must be inside the training fold.

Bad:

```text
Select top 20 features on all data → cross-validate model
```

This leaks label information and gives inflated performance.

Good:

```text
Outer CV:
    split train/test
    Inner CV:
        select features
        tune model
    Evaluate once on outer test fold
```

## 8.2 Use repeated stratified CV

Recommended:

```text
5-fold CV × 20 repeats
```

or:

```text
10-fold CV × 10 repeats
```

If class imbalance is strong, use stratification.

## 8.3 If you have batch/farm/recording-day metadata

Use grouped validation:

```text
GroupKFold
Leave-one-batch-out
Leave-one-farm-out
```

This is extremely important if recordings were collected in batches.

Otherwise, your model may learn recording/session artefacts instead of myopathy biology.

---

# 9. Literature findings relevant to your scope

## 9.1 Direct myopathy-vocalization literature

I found **no clear existing literature directly validating vocal acoustic features as biomarkers of poultry muscular myopathies**.

That means there is likely no established feature such as:

```text
MFCC_07_std is known to indicate woody breast
```

or:

```text
f0_mean is a validated biomarker of white striping
```

So your work is exploratory and potentially novel.

---

## 9.2 Related poultry vocalization literature

The closest related areas are:

### Welfare/stress monitoring

Poultry vocalizations are commonly studied as indicators of:

- stress
- fear
- isolation distress
- thermal discomfort
- general welfare

Commonly relevant acoustic variables:

```text
call rate
call duration
amplitude/energy
dominant frequency
fundamental frequency
spectral distribution
```

These overlap with your features:

```text
vocalization_rate_per_min
mean_event_duration_sec
rms_mean
rms_std
spectral_centroid_mean
spectral_rolloff_mean
f0_mean
f0_std
```

---

### Growth/body condition monitoring

Some poultry sound studies use flock vocalization features to estimate growth/body weight or age-related changes.

Relevant acoustic dimensions:

```text
dominant frequency
spectral distribution
call intensity
temporal activity
```

This matters because myopathies in broilers are often associated with growth rate, body composition, or metabolic stress. If myopathic hens also differ in body size or condition, acoustic differences may be indirect.

Potential confounders:

```text
body weight
age
breed/line
recording day
farm
environment
microphone distance
```

---

### Respiratory/disease monitoring

Sound analysis has been explored for detecting abnormal respiratory sounds or health status in poultry and livestock.

Relevant variables:

```text
spectral flatness
spectral bandwidth
spectral centroid
ZCR
MFCCs
energy distribution
```

These overlap with your:

```text
spectral_flatness_mean
spectral_bandwidth_mean
spectral_centroid_mean
zcr_mean
mfcc_* features
```

But respiratory disease is not the same as muscular myopathy. Interpretation should be cautious.

---

## 9.3 General bioacoustics literature

Broader animal-sound classification work supports the use of:

- MFCCs
- spectral features
- energy features
- temporal call features
- pretrained audio embeddings if raw audio is available

Relevant benchmark papers:

- **BEANS: The Benchmark of Animal Sounds**  
  https://huggingface.co/papers/2210.12300

- **BirdSet: A Multi-Task Benchmark for Classification in Avian Bioacoustics**  
  https://huggingface.co/papers/2403.10380

These are not myopathy-specific, but they support the general idea that acoustic features can encode biologically relevant information.

---

# 10. Features I would prioritize biologically

Given your exact feature list, I would initially focus on these.

## Highest-priority candidates

```text
silence_percent
vocalization_rate_per_min
num_high_energy_events
mean_event_duration_sec
rms_mean
rms_std
spectral_centroid_mean
spectral_bandwidth_mean
spectral_rolloff_mean
spectral_flatness_mean
zcr_mean
f0_mean
f0_std
f0_valid_ratio
```

Why these?

They are interpretable and connected to plausible behavioural/physiological mechanisms:

- amount of calling
- intensity of calling
- temporal structure
- harshness/noisiness
- pitch
- tonal reliability

---

## Medium-priority candidates

```text
spectral_centroid_std
spectral_bandwidth_std
spectral_rolloff_std
spectral_flatness_std
zcr_std
f0_min
f0_max
```

These measure variability, which may reflect unstable or heterogeneous vocal behavior.

---

## MFCC candidates

MFCCs may be predictive, but I would not manually inspect 140 MFCC statistics independently.

I would use one of these approaches:

### Option A: reduce MFCCs

Use only:

```text
mfcc_01_mean ... mfcc_20_mean
mfcc_01_std ... mfcc_20_std
```

### Option B: PCA on MFCC block

Create:

```text
MFCC_PC1
MFCC_PC2
MFCC_PC3
MFCC_PC4
MFCC_PC5
```

### Option C: stability selection

Let elastic net select stable MFCCs across folds.

---

# 11. What results would be convincing?

A convincing result would not be:

> A t-SNE plot shows two clusters.

A convincing result would be:

1. Repeated/nested cross-validation shows performance above chance.
2. Permutation test confirms the model is not exploiting random label structure.
3. Feature selection is stable.
4. Top features are biologically plausible.
5. Results survive adjustment for confounders such as body weight, age, batch, or recording conditions.
6. Similar feature families are selected across different model types.

For example, a credible conclusion would look like:

> Across repeated nested cross-validation, temporal and spectral features predicted myopathy status with ROC-AUC 0.74 ± 0.08. The most stable features were `vocalization_rate_per_min`, `silence_percent`, `spectral_flatness_mean`, and `f0_valid_ratio`. MFCC features improved performance modestly but were less stable. These results suggest that myopathic hens may differ in vocal activity and call noisiness, although external validation is required.

---

# 12. Recommended final analysis plan

If I were analyzing your data, I would do this:

## Phase 1: Exploratory analysis

- Missing values
- Outliers
- Class balance
- Correlation heatmap
- PCA plot
- Distribution plots of key features

## Phase 2: Univariate statistics

For each feature:

- group median/mean
- effect size
- p-value
- FDR-adjusted p-value
- single-feature AUC

## Phase 3: Feature-block analysis

Compare predictive value of:

1. temporal/event features
2. energy features
3. spectral features
4. F0 features
5. MFCC features
6. all features
7. PCA-reduced features

## Phase 4: Regularized modelling

Use:

- elastic-net logistic regression
- linear SVM
- random forest or gradient boosting

With:

- repeated stratified nested CV
- feature selection inside folds
- class weighting if imbalanced

## Phase 5: Interpretation

Use:

- stability selection
- permutation importance
- SHAP
- PCA loadings
- effect-size plots

## Phase 6: Sensitivity analysis

Repeat after controlling for:

- duration
- sample rate
- body weight
- age
- batch
- recording day
- microphone/session

if available.

---

# 13. Bottom-line recommendation

For your specific problem, the strongest strategy is:

> Combine interpretable univariate screening, correlation pruning, elastic-net stability selection, and block-wise PCA/MFCC reduction, all evaluated inside repeated nested cross-validation.

The most promising feature families are likely:

1. **Vocal activity**
   ```text
   vocalization_rate_per_min
   silence_percent
   num_high_energy_events
   mean_event_duration_sec
   ```

2. **Spectral/noisiness features**
   ```text
   spectral_centroid_mean
   spectral_bandwidth_mean
   spectral_rolloff_mean
   spectral_flatness_mean
   zcr_mean
   ```

3. **Pitch/F0 reliability**
   ```text
   f0_mean
   f0_std
   f0_valid_ratio
   ```

4. **Reduced MFCC/timbre features**
   ```text
   MFCC PCA components or stable MFCC coefficients
   ```

The most important caution is that **there is no established direct myopathy-vocalization biomarker literature**, so the study should be presented as exploratory and validated carefully.