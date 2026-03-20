###############################################################################

# Remove all variables
# import sys
# sys.modules[__name__].__dict__.clear()

# Import libraries
import os
import numpy as np
import pandas as pd
import glob

# Libraries for ML
import matplotlib.pyplot as plt
import seaborn as sns

    
    
# Set directory
os.chdir("/home/rj/ownCloud/PROJECTS/MilanBrazdilMetylace/SCIKIT_ML_BRAZDIL+KOBOW_V2/HYPERPARAMETER_TUNING/")
print(os.getcwd())



comparison_df = pd.read_csv("feature_set_comparison_metrics.csv")
comparison_df = comparison_df.sort_values('n_features')

x = comparison_df['n_features'].values

fig, ax = plt.subplots(figsize=(12, 5))

# Balanced Accuracy
ax.plot(x, comparison_df['cv_balanced_accuracy_mean'], 
        marker='o', linewidth=2, markersize=7,
        color='#2E75B6', label='Balanced Accuracy')
ax.fill_between(x,
                comparison_df['cv_balanced_accuracy_mean'] - comparison_df['cv_balanced_accuracy_std'],
                comparison_df['cv_balanced_accuracy_mean'] + comparison_df['cv_balanced_accuracy_std'],
                alpha=0.2, color='#2E75B6')

# F1 Score
ax.plot(x, comparison_df['cv_f1_mean'],
        marker='s', linewidth=2, markersize=7,
        color='#E8702A', label='F1 Score (weighted)')
ax.fill_between(x,
                comparison_df['cv_f1_mean'] - comparison_df['cv_f1_std'],
                comparison_df['cv_f1_mean'] + comparison_df['cv_f1_std'],
                alpha=0.2, color='#E8702A')

# ROC AUC
ax.plot(x, comparison_df['cv_roc_auc_mean'],
        marker='^', linewidth=2, markersize=7,
        color='#2CA02C', label='ROC AUC')
ax.fill_between(x,
                comparison_df['cv_roc_auc_mean'] - comparison_df['cv_roc_auc_std'],
                comparison_df['cv_roc_auc_mean'] + comparison_df['cv_roc_auc_std'],
                alpha=0.2, color='#2CA02C')

# Highlight best
best_n = 8000
ax.axvline(best_n, color='goldenrod', linestyle='--', linewidth=1.5, alpha=0.8)
ax.axvspan(best_n - 500, best_n + 500, color='gold', alpha=0.15, label='Best (N=8,000)')

ax.set_xlabel('Number of Selected Features (N)', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Classification Performance vs. Number of Selected Features\n'
             '(Nested CV mean ± 1 SD, 5 outer folds)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"{int(n):,}" for n in x], rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig("feature_count_lineplot.svg", format="svg", bbox_inches='tight')
plt.savefig("feature_count_lineplot.png", dpi=300, bbox_inches='tight')
plt.show()