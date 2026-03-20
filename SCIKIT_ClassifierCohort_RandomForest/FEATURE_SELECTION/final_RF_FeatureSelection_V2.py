###############################################################################

# Remove all variables
# import sys
# sys.modules[__name__].__dict__.clear()

# Import libraries
import os
import numpy as np
import pandas as pd

# Set directory
os.chdir("/home/rj/ownCloud/PROJECTS/MilanBrazdilMetylace/SCIKIT_ML_BRAZDIL+KOBOW_V2/FEATURE_SELECTION/")
print(os.getcwd())

topX = 10000 #select topX features
################################################################################
# LOAD DATA

data_path = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort_February2026_imputed/FCD_sesame_BETAvalues_February2026_removedBatch_forPython.tsv"
meta_path = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort_February2026_imputed/FCD_EPIC_SAMPLES_February2026_forPython_TRAIN.tsv"

data_df = pd.read_csv(data_path, sep='\t')
meta_df = pd.read_csv(meta_path, sep='\t')


#set index to be one of columns but keep the original
data_df = data_df.set_index("Sample", drop = False)
meta_df = meta_df.set_index("Sample", drop = False)

# Make sure sample IDs align between data and metadata
intersection = data_df.index.intersection(meta_df.index)
print(f" Matching sample IDs: {len(intersection)} / {len(meta_df.index)}")

# Find mismatches
missing_in_data = meta_df.index.difference(data_df.index)
missing_in_meta = data_df.index.difference(meta_df.index)

if len(missing_in_data) > 0:
    print("\n Missing in data_df:", missing_in_data[:10])
if len(missing_in_meta) > 0:
    print("\n Missing in meta_df:", missing_in_meta[:10])


# Keep only shared samples and ensure same order
shared_samples = data_df.index.intersection(meta_df.index)
data_df = data_df.loc[shared_samples]
meta_df = meta_df.loc[shared_samples]

print(meta_df.index)
print(data_df['Sample'])
print(meta_df['Sample'])
print(meta_df['SUBTYPE'])
print(meta_df['SUBTYPE_NUM'])


set(data_df.index) == set(meta_df.index)


###############################################################################
# Align samples and prepare features/target

# Example: encode a categorical column
X = data_df.drop('Sample', axis=1)
y = meta_df["SUBTYPE_NUM"]  # or whichever metadata column is your target
print(y.unique())
print(X.head)




###############################################################################
###############################################################################
# RANDOM FOREST FEATURE SELECTION - ENHANCED WITH COMPREHENSIVE METRICS
# Based on Nature paper methodology with additional statistical analysis


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (balanced_accuracy_score, classification_report, 
                             accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, matthews_corrcoef, cohen_kappa_score)
from collections import Counter
from scipy import stats
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Assuming X, y, and feature_names are defined
# If feature_names doesn't exist, create it
if 'feature_names' not in locals():
    if hasattr(X, 'columns'):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

# Convert to numpy arrays for faster processing
X_array = X.values if hasattr(X, 'values') else X
y_array = y.values if hasattr(y, 'values') else y

# -----------------------------------------------------------------------------
# Step 1: Examine class distribution
# -----------------------------------------------------------------------------
print("="*80)
print("CLASS DISTRIBUTION ANALYSIS")
print("="*80)

class_counts = Counter(y_array)
for class_label, count in sorted(class_counts.items()):
    percentage = (count / len(y_array)) * 100
    print(f"  Class {class_label}: {count:>6} samples ({percentage:>5.2f}%)")

n_classes = len(class_counts)
min_class_size = min(class_counts.values())
max_class_size = max(class_counts.values())
imbalance_ratio = max_class_size / min_class_size

print(f"\nTotal classes: {n_classes}")
print(f"Minimum class size: {min_class_size}")
print(f"Maximum class size: {max_class_size}")
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

# -----------------------------------------------------------------------------
# Step 2: Calculate parameters
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("FEATURE SELECTION PARAMETERS")
print("="*80)

n_total_features = X_array.shape[1]
n_samples = X_array.shape[0]

# max_features parameter (mtry)
mtry_max_features = int(np.sqrt(n_total_features))

# Downsampling: use minimum class size
downsample_size = min_class_size

# Number of iterations for stable importance estimates
n_iterations = 100  # Adjust as needed
n_trees_per_iteration = 100  # Adjust based on computational resources

print(f"Total features: {n_total_features:,}")
print(f"Total samples: {n_samples:,}")
print(f"mtry (max_features): {mtry_max_features:,}")
print(f"Downsample size per class: {downsample_size}")
print(f"Iterations: {n_iterations}")
print(f"Trees per iteration: {n_trees_per_iteration}")
print(f"Total trees: {n_iterations * n_trees_per_iteration:,}")

# -----------------------------------------------------------------------------
# Step 3: Feature Selection with Enhanced Metrics
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("FEATURE SELECTION WITH METRICS TRACKING")
print("="*80)

# Initialize accumulators
accumulated_importances = np.zeros(n_total_features)
importance_per_iteration = []

# Track Out-of-Bag scores and other metrics
oob_scores = []
balanced_acc_scores = []
f1_scores = []

# For tracking feature stability
feature_rank_per_iteration = []

print("\nRunning feature selection iterations...")
for iter_idx in range(n_iterations):
    if (iter_idx + 1) % 10 == 0 or iter_idx == 0:
        print(f"  Iteration {iter_idx + 1}/{n_iterations}...")
    
    # ---------------------------------------------------------------------
    # Manual downsampling - sample equal number from each class
    # ---------------------------------------------------------------------
    sample_indices = []
    for class_label in class_counts.keys():
        class_indices = np.where(y_array == class_label)[0]
        
        # Sample with replacement if class is smaller than downsample_size
        sampled = np.random.choice(
            class_indices,
            size=min(downsample_size, len(class_indices)),
            replace=len(class_indices) < downsample_size
        )
        sample_indices.extend(sampled)
    
    sample_indices = np.array(sample_indices)
    np.random.shuffle(sample_indices)
    
    # Get downsampled data
    X_train_sample = X_array[sample_indices]
    y_train_sample = y_array[sample_indices]
    
    # ---------------------------------------------------------------------
    # Train Random Forest with OOB scoring
    # ---------------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=n_trees_per_iteration,
        max_features=mtry_max_features,
        n_jobs=-1,
        random_state=42 + iter_idx,
        bootstrap=True,
        oob_score=True  # Enable OOB scoring
    )
    
    rf.fit(X_train_sample, y_train_sample)
    
    # Store importances
    accumulated_importances += rf.feature_importances_
    importance_per_iteration.append(rf.feature_importances_.copy())
    
    # Track OOB score
    oob_scores.append(rf.oob_score_)
    
    # Evaluate on held-out validation set (10% of original data)
    if iter_idx == 0:
        # Create a single held-out validation set
        X_train_full, X_val, y_train_full, y_val = train_test_split(
            X_array, y_array, test_size=0.1, random_state=42, stratify=y_array
        )
    
    # Predict on validation set
    y_pred = rf.predict(X_val)
    balanced_acc_scores.append(balanced_accuracy_score(y_val, y_pred))
    
    if n_classes == 2:
        f1_scores.append(f1_score(y_val, y_pred))
    else:
        f1_scores.append(f1_score(y_val, y_pred, average='weighted'))
    
    # Track feature rankings
    ranks = stats.rankdata(-rf.feature_importances_)  # Negative for descending
    feature_rank_per_iteration.append(ranks)

print(f"\n✓ Feature selection complete!")

# -----------------------------------------------------------------------------
# Step 4: Calculate Comprehensive Metrics
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("CALCULATING COMPREHENSIVE FEATURE METRICS")
print("="*80)

# Average importances across all iterations
averaged_importances = accumulated_importances / n_iterations

# Convert to numpy array for calculations
importance_matrix = np.array(importance_per_iteration)
rank_matrix = np.array(feature_rank_per_iteration)

# Calculate statistics for each feature
feature_metrics = []

for i, feat_name in enumerate(feature_names):
    importances_for_feature = importance_matrix[:, i]
    ranks_for_feature = rank_matrix[:, i]
    
    metrics = {
        'feature': feat_name,
        
        # Basic importance statistics
        'importance_mean': averaged_importances[i],
        'importance_std': np.std(importances_for_feature),
        'importance_median': np.median(importances_for_feature),
        'importance_min': np.min(importances_for_feature),
        'importance_max': np.max(importances_for_feature),
        'importance_range': np.max(importances_for_feature) - np.min(importances_for_feature),
        
        # Importance quartiles
        'importance_q25': np.percentile(importances_for_feature, 25),
        'importance_q75': np.percentile(importances_for_feature, 75),
        'importance_iqr': np.percentile(importances_for_feature, 75) - np.percentile(importances_for_feature, 25),
        
        # Coefficient of variation (stability measure)
        'importance_cv': np.std(importances_for_feature) / (averaged_importances[i] + 1e-10),
        'stability_score': 1 - (np.std(importances_for_feature) / (averaged_importances[i] + 1e-10)),
        
        # Ranking statistics
        'rank_mean': np.mean(ranks_for_feature),
        'rank_std': np.std(ranks_for_feature),
        'rank_median': np.median(ranks_for_feature),
        'rank_min': np.min(ranks_for_feature),
        'rank_max': np.max(ranks_for_feature),
        
        # Selection frequency (how often feature ranks in top 1%, 5%, 10%)
        'top_1pct_frequency': np.mean(ranks_for_feature <= n_total_features * 0.01) * 100,
        'top_5pct_frequency': np.mean(ranks_for_feature <= n_total_features * 0.05) * 100,
        'top_10pct_frequency': np.mean(ranks_for_feature <= n_total_features * 0.10) * 100,
        
        # Consistency (how often importance is above mean)
        'above_mean_frequency': np.mean(importances_for_feature > averaged_importances[i]) * 100,
        
        # Z-score of mean importance
        'importance_zscore': (averaged_importances[i] - np.mean(averaged_importances)) / (np.std(averaged_importances) + 1e-10),
    }
    
    feature_metrics.append(metrics)

# Create DataFrame
feature_metrics_df = pd.DataFrame(feature_metrics)

# Add overall rank
feature_metrics_df['overall_rank'] = feature_metrics_df['importance_mean'].rank(ascending=False)
feature_metrics_df = feature_metrics_df.sort_values('importance_mean', ascending=False)

# -----------------------------------------------------------------------------
# Step 5: Model Performance Metrics
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("MODEL PERFORMANCE METRICS")
print("="*80)

print(f"\nOut-of-Bag (OOB) Performance across {n_iterations} iterations:")
print(f"  Mean OOB Score: {np.mean(oob_scores):.4f} ± {np.std(oob_scores):.4f}")
print(f"  Median OOB Score: {np.median(oob_scores):.4f}")
print(f"  Min OOB Score: {np.min(oob_scores):.4f}")
print(f"  Max OOB Score: {np.max(oob_scores):.4f}")

print(f"\nValidation Set Performance:")
print(f"  Mean Balanced Accuracy: {np.mean(balanced_acc_scores):.4f} ± {np.std(balanced_acc_scores):.4f}")
print(f"  Mean F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

# -----------------------------------------------------------------------------
# Step 6: Feature Importance Distribution
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("FEATURE IMPORTANCE DISTRIBUTION")
print("="*80)

print(f"\nImportance score statistics (across all {n_total_features:,} features):")
print(f"  Mean: {np.mean(averaged_importances):.8f}")
print(f"  Median: {np.median(averaged_importances):.8f}")
print(f"  Std Dev: {np.std(averaged_importances):.8f}")
print(f"  Min: {np.min(averaged_importances):.8f}")
print(f"  Max: {np.max(averaged_importances):.8f}")
print(f"  25th percentile: {np.percentile(averaged_importances, 25):.8f}")
print(f"  75th percentile: {np.percentile(averaged_importances, 75):.8f}")

# Concentration metrics
cumsum_importance = np.cumsum(np.sort(averaged_importances)[::-1])
total_importance = np.sum(averaged_importances)
n_top_50pct = np.argmax(cumsum_importance >= total_importance * 0.5) + 1
n_top_80pct = np.argmax(cumsum_importance >= total_importance * 0.8) + 1
n_top_90pct = np.argmax(cumsum_importance >= total_importance * 0.9) + 1

print(f"\nImportance concentration:")
print(f"  Top {n_top_50pct:,} features capture 50% of total importance")
print(f"  Top {n_top_80pct:,} features capture 80% of total importance")
print(f"  Top {n_top_90pct:,} features capture 90% of total importance")

# -----------------------------------------------------------------------------
# Step 7: Display Top Features
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("TOP 20 MOST IMPORTANT FEATURES")
print("="*80)

display_cols = ['feature', 'importance_mean', 'importance_std', 'stability_score', 
                'rank_mean', 'top_1pct_frequency', 'top_5pct_frequency']
print("\n" + feature_metrics_df[display_cols].head(20).to_string(index=False))

# -----------------------------------------------------------------------------
# Step 8: Feature Selection - Top N by Importance (Incremental)
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("FEATURE SELECTION - TOP N BY IMPORTANCE")
print("="*80)

increment_start = 1000
increment_step  = 1000
increment_end   = 20000  # e.g. gives: 500, 1500

print("\nFeature counts per threshold:")
print("-" * 40)
for n in range(increment_start, increment_end + 1, increment_step):
    n_capped = min(n, len(feature_metrics_df))
    print(f"  Top-{n_capped}: {n_capped:,} features")

# -----------------------------------------------------------------------------
# Step 9: Save Results
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

feature_metrics_df.to_csv("feature_metrics_comprehensive.csv", index=False)
print("✓ Saved: feature_metrics_comprehensive.csv")

for n in range(increment_start, increment_end + 1, increment_step):
    n_capped = min(n, len(feature_metrics_df))
    subset_df = feature_metrics_df.head(n_capped).copy()
    fname = f"selected_features_top{n_capped}.csv"
    subset_df.to_csv(fname, index=False)
    print(f"✓ Saved: {fname}  ({n_capped:,} features)")

performance_metrics = pd.DataFrame({
    'iteration': range(1, n_iterations + 1),
    'oob_score': oob_scores,
    'balanced_accuracy': balanced_acc_scores,
    'f1_score': f1_scores
})
performance_metrics.to_csv("model_performance_per_iteration.csv", index=False)
print("✓ Saved: model_performance_per_iteration.csv")