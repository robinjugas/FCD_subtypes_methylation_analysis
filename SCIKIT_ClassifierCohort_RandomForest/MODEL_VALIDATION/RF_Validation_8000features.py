################################################################################
# VALIDATE TRAINED MODEL ON INDEPENDENT COHORT
################################################################################

from sklearn.preprocessing import label_binarize
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

# Set directory
os.chdir("/home/rj/ownCloud/PROJECTS/MilanBrazdilMetylace/SCIKIT_ML_BRAZDIL+KOBOW_V2/MODEL_VALIDATION/")

################################################################################
# 1. LOAD TRAINED MODEL
################################################################################
print("="*80)
print("LOADING TRAINED MODEL")
print("="*80)

# Load the RFmodel
RFmodel = joblib.load('/home/rj/ownCloud/PROJECTS/MilanBrazdilMetylace/SCIKIT_ML_BRAZDIL+KOBOW_V2/HYPERPARAMETER_TUNING/features_top8000/random_forest_final_model.pkl')

# Load feature information
feature_info = joblib.load('/home/rj/ownCloud/PROJECTS/MilanBrazdilMetylace/SCIKIT_ML_BRAZDIL+KOBOW_V2/HYPERPARAMETER_TUNING/features_top8000/random_forest_model_with_metadata.pkl')
selected_features = feature_info['feature_names']
hyperparameters = feature_info['hyperparameters']

print(f"Model loaded successfully!")
print(f"Number of features used: {len(selected_features)}")

################################################################################
# 2. LOAD VALIDATION DATA
################################################################################
print("\n" + "="*80)
print("LOADING VALIDATION DATA")
print("="*80)

# Update these paths to your validation cohort
validation_data_path = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort_February2026_imputed/FCD_sesame_BETAvalues_February2026_removedBatch_forPython.tsv"
validation_meta_path = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort_February2026_imputed/FCD_EPIC_SAMPLES_February2026_forPython_VALIDATION.tsv"



validation_data_df = pd.read_csv(validation_data_path, sep='\t')
validation_meta_df = pd.read_csv(validation_meta_path, sep='\t')

# Set index
validation_data_df = validation_data_df.set_index("Sample", drop=False)
validation_meta_df = validation_meta_df.set_index("Sample", drop=False)

# Align samples
shared_samples = validation_data_df.index.intersection(validation_meta_df.index)
validation_data_df = validation_data_df.loc[shared_samples]
validation_meta_df = validation_meta_df.loc[shared_samples]

print(f"Validation cohort: {len(shared_samples)} samples")


################################################################################


################################################################################
# 3. PREPARE VALIDATION DATA
################################################################################
# Get both numeric and string labels
y_val_numeric = validation_meta_df["SUBTYPE_NUM"]
y_val_string = validation_meta_df["SUBTYPE"]

# Create mapping from numeric to string labels
label_mapping = dict(zip(y_val_numeric, y_val_string))
print(f"\nLabel mapping: {label_mapping}")

# Use numeric labels for prediction (model expects these)
X_val = validation_data_df.drop('Sample', axis=1)
X_val = X_val[selected_features]
X_val_array = X_val.values
y_val_array = y_val_numeric.values

print(f"\nValidation features shape: {X_val_array.shape}")
print(f"Validation labels shape: {y_val_array.shape}")
print(f"Validation class distribution:\n{pd.Series(y_val_array).value_counts()}")

################################################################################
# 4. MAKE PREDICTIONS
################################################################################
print("\n" + "="*80)
print("MAKING PREDICTIONS ON VALIDATION COHORT")
print("="*80)

# Predict (returns numeric labels)
y_val_pred = RFmodel.predict(X_val_array)
y_val_proba = RFmodel.predict_proba(X_val_array)

# Convert to string labels for interpretability
y_val_true_str = pd.Series(y_val_array).map(label_mapping).values
y_val_pred_str = pd.Series(y_val_pred).map(label_mapping).values

# Calculate metrics (use string labels from here on)
val_accuracy = accuracy_score(y_val_true_str, y_val_pred_str)
print(f"\nValidation Accuracy: {val_accuracy:.4f}")

################################################################################
# 5. DETAILED EVALUATION
################################################################################
print("\n" + "="*80)
print("VALIDATION COHORT EVALUATION")
print("="*80)

# Classification report (with string labels)
print("\nClassification Report:")
print(classification_report(y_val_true_str, y_val_pred_str))

# Confusion matrix (with string labels)
cm = confusion_matrix(y_val_true_str, y_val_pred_str)
unique_labels = sorted(np.unique(y_val_true_str))

# Per-class accuracy (with string labels)
print("\nPer-class accuracy:")
for cls in unique_labels:
    cls_mask = (y_val_true_str == cls)
    cls_acc = np.mean(y_val_pred_str[cls_mask] == y_val_true_str[cls_mask])
    print(f"  {cls}: {cls_acc:.4f} ({np.sum(cls_mask)} samples)")

###############################################################################

fig = plt.figure(figsize=(10, 4))
gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)


################################################################################
# 6. ROC CURVES (for binary or multi-class)
################################################################################

# Use string labels for display
unique_labels = sorted(np.unique(y_val_true_str))
n_classes = len(unique_labels)

# Multi-class: One-vs-Rest ROC curves
y_val_bin = label_binarize(y_val_true_str, classes=unique_labels)

colors = plt.colormaps.get_cmap('tab10')(np.linspace(0, 1, n_classes))


ax2 = fig.add_subplot(gs[0, 0])

for i, string_label in enumerate(unique_labels):
     # Find corresponding numeric label and probability column index
     numeric_label = [k for k, v in label_mapping.items()
                      if v == string_label][0]
     prob_idx = list(sorted(label_mapping.keys())).index(numeric_label)

     fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_val_proba[:, prob_idx])
     auc = roc_auc_score(y_val_bin[:, i], y_val_proba[:, prob_idx])
     ax2.plot(fpr, tpr, color=colors[i], lw=3,
              label=f'{string_label} (AUC={auc:.3f})')


ax2.plot([0, 1], [0, 1], 'k--', label='Random')
ax2.set_xlabel('False Positive Rate', fontsize=8)
ax2.set_ylabel('True Positive Rate', fontsize=8)
ax2.set_title('a) ROC Curves - Test set (One-vs-Rest)',
          fontsize=12, fontweight='normal')
ax2.legend(fontsize=8, loc='lower right')
ax2.grid(True, alpha=0.3)


################################################################################
# 6. Confusion matrix
################################################################################
ax1 = fig.add_subplot(gs[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax1,
            xticklabels=unique_labels, yticklabels=unique_labels)
ax1.set_title('b) Confusion Matrix - Test set',fontsize=12, fontweight='normal')
ax1.set_ylabel('True Label', fontsize=8)
ax1.set_xlabel('Predicted Label', fontsize=8)
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=45, va='center')

# ax1.legend(fontsize=8) #, loc='lower right'
# plt.tight_layout()




plt.savefig("random_forest_TEST_set_FINAL.svg", format="svg",bbox_inches='tight')
plt.show()


################################################################################
################################################################################
# 7. SAVE VALIDATION RESULTS
################################################################################
# Save predictions with probabilities (using string labels)
validation_results = pd.DataFrame({
    'Sample': validation_meta_df['Sample'].values,
    'True_Label': y_val_true_str,
    'Predicted_Label': y_val_pred_str,
    'Correct': y_val_true_str == y_val_pred_str
})

# Add probability columns using string class names
for i, numeric_label in enumerate(sorted(label_mapping.keys())):
    string_label = label_mapping[numeric_label]
    validation_results[f'Probability_{string_label}'] = y_val_proba[:, i]

validation_results.to_csv('validation_predictions.csv', index=False)

# Save summary statistics
summary_stats = {
    'validation_accuracy': val_accuracy,
    'n_samples': len(y_val_true_str),
    'class_distribution': pd.Series(y_val_true_str).value_counts().to_dict()
}

with open('validation_summary.txt', 'w') as f:
    f.write("VALIDATION COHORT SUMMARY\n")
    f.write("="*80 + "\n")
    f.write(f"Accuracy: {val_accuracy:.4f}\n")
    f.write(f"N samples: {len(y_val_true_str)}\n")
    f.write(f"\nClass distribution:\n{pd.Series(y_val_true_str).value_counts()}\n")
    f.write(f"\nClassification Report:\n")
    f.write(classification_report(y_val_true_str, y_val_pred_str))
    f.write(f"\nConfusion Matrix:\n{cm}\n")