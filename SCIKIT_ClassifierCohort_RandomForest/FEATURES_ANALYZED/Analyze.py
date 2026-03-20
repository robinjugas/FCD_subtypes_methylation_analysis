###############################################################################
# FEATURE IMPORTANCE PER CLASS ANALYSIS
# Loads trained Random Forest model and analyzes which features are associated
# with each subtype/predicted class
###############################################################################

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Set output directory
os.chdir("/home/rj/ownCloud/PROJECTS/MilanBrazdilMetylace/SCIKIT_ML_BRAZDIL+KOBOW_V2/FEATURES_ANALYZED/")
print(os.getcwd())

###############################################################################
# 1. LOAD TRAINED MODEL AND METADATA
###############################################################################

# Load model metadata (contains model, feature names, label mapping, etc.)
metadata_filename = "/home/rj/ownCloud/PROJECTS/MilanBrazdilMetylace/SCIKIT_ML_BRAZDIL+KOBOW_V2/HYPERPARAMETER_TUNING/features_top8000/random_forest_model_with_metadata.pkl"
model_metadata = joblib.load(metadata_filename)

# Extract everything we need
final_model       = model_metadata['model']
feature_names_selected = model_metadata['feature_names']
label_mapping     = model_metadata['label_mapping']
n_features        = model_metadata['n_features']
n_classes         = model_metadata['n_classes']

print(f"✓ Model loaded successfully!")
print(f"  Features:          {n_features}")
print(f"  Classes:           {n_classes}")
print(f"  Label mapping:     {label_mapping}")
print(f"  n_estimators:      {final_model.n_estimators}")
print(f"  OOB Score:         {final_model.oob_score_:.4f}")


###############################################################################
# 2. LOAD TRAINING DATA (needed for per-class feature value analysis)
###############################################################################

print("\n" + "="*80)
print("LOADING TRAINING DATA")
print("="*80)

data_path = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort_February2026_imputed/FCD_sesame_BETAvalues_February2026_removedBatch_forPython.tsv"
meta_path = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort_February2026_imputed/FCD_EPIC_SAMPLES_February2026_forPython_TRAIN.tsv"

data_df = pd.read_csv(data_path, sep='\t')
meta_df = pd.read_csv(meta_path, sep='\t')

# Set index
data_df = data_df.set_index("Sample", drop=False)
meta_df = meta_df.set_index("Sample", drop=False)

# Keep only shared samples
shared_samples = data_df.index.intersection(meta_df.index)
data_df = data_df.loc[shared_samples]
meta_df = meta_df.loc[shared_samples]

print(f"✓ Data loaded: {len(shared_samples)} samples")

# Prepare X and y
X = data_df.drop('Sample', axis=1)
y_numeric = meta_df["SUBTYPE_NUM"].values
y_string  = meta_df["SUBTYPE"].values

# Select only the model's features in the correct order
X_selected = X[feature_names_selected]
X_array    = X_selected.values

print(f"✓ Feature matrix shape: {X_array.shape}")
print(f"  Class distribution: {dict(zip(*np.unique(y_numeric, return_counts=True)))}")

###############################################################################
# 3. CALCULATE MEAN FEATURE VALUES PER CLASS
###############################################################################

print("\n" + "="*80)
print("CALCULATING FEATURE STATISTICS PER CLASS")
print("="*80)

feature_values_per_class = {}
for numeric_label in sorted(np.unique(y_numeric)):
    string_label = label_mapping[numeric_label]
    mask = (y_numeric == numeric_label)
    class_samples = X_array[mask]

    feature_values_per_class[string_label] = {
        'mean':      np.mean(class_samples, axis=0),
        'std':       np.std(class_samples, axis=0),
        'median':    np.median(class_samples, axis=0),
        'n_samples': int(np.sum(mask))
    }

print(f"✓ Calculated feature statistics for {len(feature_values_per_class)} classes:")
for label, stats in feature_values_per_class.items():
    print(f"  {label}: {stats['n_samples']} samples")

class_labels = sorted(feature_values_per_class.keys())

###############################################################################
# 4. IDENTIFY TOP DISCRIMINATIVE FEATURES PER CLASS
###############################################################################

print("\n" + "="*80)
print("IDENTIFYING DISCRIMINATIVE FEATURES PER CLASS")
print("="*80)

top_n_features = 1000  # Number of top features to report per class
discriminative_features_per_class = {}

for string_label, stats in feature_values_per_class.items():
    class_mean = stats['mean']

    # Mean across all other classes
    other_classes_mean = np.mean(
        [feature_values_per_class[other]['mean']
         for other in feature_values_per_class if other != string_label],
        axis=0
    )

    # Raw difference from other classes
    difference = class_mean - other_classes_mean

    # Weight difference by overall feature importance
    weighted_score = difference * final_model.feature_importances_

    # Top features with highest weighted score (most discriminative upwards)
    top_up_indices   = np.argsort(weighted_score)[-top_n_features:][::-1]
    # Top features with lowest weighted score (most discriminative downwards)
    top_down_indices = np.argsort(weighted_score)[:top_n_features]

    rows = []
    for direction, indices in [('higher_in_class', top_up_indices),
                                ('lower_in_class',  top_down_indices)]:
        for idx in indices:
            rows.append({
                'feature':        feature_names_selected[idx],
                'direction':      direction,
                'importance':     final_model.feature_importances_[idx],
                'mean_in_class':  class_mean[idx],
                'mean_in_others': other_classes_mean[idx],
                'difference':     difference[idx],
                'weighted_score': weighted_score[idx]
            })

    discriminative_features_per_class[string_label] = rows

print(f"✓ Top {top_n_features} discriminative features (up + down) identified per class")

###############################################################################
# 5. PRINT RESULTS TO CONSOLE
###############################################################################

print("\n" + "="*80)
print("TOP DISCRIMINATIVE FEATURES PER CLASS")
print("="*80)

for string_label in sorted(discriminative_features_per_class.keys()):
    n_samples = feature_values_per_class[string_label]['n_samples']
    rows = discriminative_features_per_class[string_label]

    higher = [r for r in rows if r['direction'] == 'higher_in_class']
    lower  = [r for r in rows if r['direction'] == 'lower_in_class']

    print(f"\n{'='*80}")
    print(f"  {string_label}   (n = {n_samples} samples)")
    print(f"{'='*80}")

    print(f"\n  ▲ Features HIGHER in {string_label} vs other classes:")
    print(f"  {'Feature':<35} {'Importance':>10}  {'Mean(class)':>11}  {'Mean(others)':>12}  {'Diff':>8}")
    print(f"  {'-'*78}")
    for r in higher[:10]:
        print(f"  {r['feature'][:34]:<35} {r['importance']:>10.6f}  "
              f"{r['mean_in_class']:>11.4f}  {r['mean_in_others']:>12.4f}  {r['difference']:>8.4f}")

    print(f"\n  ▼ Features LOWER in {string_label} vs other classes:")
    print(f"  {'Feature':<35} {'Importance':>10}  {'Mean(class)':>11}  {'Mean(others)':>12}  {'Diff':>8}")
    print(f"  {'-'*78}")
    for r in lower[:10]:
        print(f"  {r['feature'][:34]:<35} {r['importance']:>10.6f}  "
              f"{r['mean_in_class']:>11.4f}  {r['mean_in_others']:>12.4f}  {r['difference']:>8.4f}")

###############################################################################
# 6. SAVE PER-CLASS CSV FILES
###############################################################################

print("\n" + "="*80)
print("SAVING PER-CLASS FEATURE FILES")
print("="*80)

for string_label, rows in discriminative_features_per_class.items():
    df = pd.DataFrame(rows)
    filename = f"discriminative_features_{string_label.replace('/', '_')}.csv"
    df.to_csv(filename, index=False)
    print(f"✓ Saved {filename}")

###############################################################################
# 7. HEATMAP: Top overall features across all classes (Z-score)
###############################################################################

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

top_overall_n = 30
top_overall_indices = np.argsort(final_model.feature_importances_)[-top_overall_n:][::-1]
top_overall_features = [feature_names_selected[i] for i in top_overall_indices]

# Build Z-score matrix: rows = features, columns = classes
heatmap_data = np.zeros((len(top_overall_features), len(class_labels)))

for i, feat_idx in enumerate(top_overall_indices):
    all_means = np.array([feature_values_per_class[cls]['mean'][feat_idx]
                          for cls in class_labels])
    mean_all = np.mean(all_means)
    std_all  = np.std(all_means)
    for j, cls in enumerate(class_labels):
        val = feature_values_per_class[cls]['mean'][feat_idx]
        heatmap_data[i, j] = (val - mean_all) / std_all if std_all > 0 else 0

# Plot
fig, ax = plt.subplots(figsize=(12, 16))
sns.heatmap(heatmap_data,
            xticklabels=class_labels,
            yticklabels=top_overall_features,
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Z-score (relative to mean across classes)'},
            ax=ax,
            linewidths=0.5)
ax.set_xlabel('Subtype', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature (CpG)', fontsize=12, fontweight='bold')
ax.set_title(f'Top {top_overall_n} Most Important Features Across Subtypes\n'
             f'(Z-score normalized mean methylation beta values)',
             fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig('feature_importance_per_class_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved feature_importance_per_class_heatmap.png")

# Save heatmap data
heatmap_df = pd.DataFrame(heatmap_data,
                           index=top_overall_features,
                           columns=class_labels)
heatmap_df.to_csv('feature_zscore_per_class_heatmap_data.csv')
print("✓ Saved feature_zscore_per_class_heatmap_data.csv")

###############################################################################
# 8. BARPLOT: Top discriminative features for each class individually
###############################################################################

n_cols = 2
n_rows = int(np.ceil(len(class_labels) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
axes = axes.flatten()

for ax_idx, string_label in enumerate(sorted(class_labels)):
    ax = axes[ax_idx]
    rows = discriminative_features_per_class[string_label]

    # Top 10 up + top 10 down, sorted by absolute weighted score
    plot_rows = sorted(rows, key=lambda r: abs(r['weighted_score']), reverse=True)[:15]
    plot_rows = sorted(plot_rows, key=lambda r: r['weighted_score'])

    features   = [r['feature'][:35] for r in plot_rows]
    scores     = [r['weighted_score'] for r in plot_rows]
    colors     = ['#d73027' if s > 0 else '#4575b4' for s in scores]

    y_pos = np.arange(len(features))
    ax.barh(y_pos, scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=8)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Weighted Discriminative Score\n(importance × mean difference)', fontsize=9)
    ax.set_title(f'{string_label}\n(n={feature_values_per_class[string_label]["n_samples"]} samples)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

# Hide unused subplots
for ax_idx in range(len(class_labels), len(axes)):
    axes[ax_idx].set_visible(False)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#d73027', label='Higher in class vs others'),
                   Patch(facecolor='#4575b4', label='Lower in class vs others')]
fig.legend(handles=legend_elements, loc='lower right',
           fontsize=11, framealpha=0.9)

plt.suptitle('Top Discriminative Features per Subtype\n'
             '(weighted by feature importance × mean difference)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('discriminative_features_per_class_barplots.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved discriminative_features_per_class_barplots.png")

###############################################################################
# 9. SUMMARY
###############################################################################

print("\n" + "="*80)
print("FEATURE IMPORTANCE PER CLASS ANALYSIS COMPLETE")
print("="*80)
print("\nFiles saved:")
for string_label in sorted(class_labels):
    filename = f"discriminative_features_{string_label.replace('/', '_')}.csv"
    print(f"  • {filename}")
print(f"  • feature_importance_per_class_heatmap.png")
print(f"  • feature_zscore_per_class_heatmap_data.csv")
print(f"  • discriminative_features_per_class_barplots.png")
print("="*80)