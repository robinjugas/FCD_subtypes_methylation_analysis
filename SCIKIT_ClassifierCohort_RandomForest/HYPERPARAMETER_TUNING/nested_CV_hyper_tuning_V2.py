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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from scipy.stats import randint, uniform
from sklearn.utils.parallel import Parallel, delayed
import joblib
from collections import defaultdict
    
    
# Set directory
os.chdir("/home/rj/ownCloud/PROJECTS/MilanBrazdilMetylace/SCIKIT_ML_BRAZDIL+KOBOW_V2/HYPERPARAMETER_TUNING/")
print(os.getcwd())


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
y_numeric = meta_df["SUBTYPE_NUM"]  # Numeric for model training
y_string = meta_df["SUBTYPE"]  # String for display

# Create label mapping
label_mapping = dict(zip(y_numeric, y_string))
print(f"\nLabel mapping: {label_mapping}")

y = y_numeric  # Use numeric for training
print(y.unique())
print(X.head)


labels = meta_df["Sample"]


# Store original column names before any processing, contains features
if hasattr(X, 'columns'):
    original_column_names = X.columns.tolist()
else:
    original_column_names = [f"Feature_{i}" for i in range(X.shape[1])]

print(original_column_names[1:10])


# Convert to numpy array for faster computation
if hasattr(X, 'values'):
    X_array = X.values
else:
    X_array = X

if hasattr(y, 'values'):
    y_array = y.values
else:
    y_array = y
    
# Get feature names
if hasattr(X, 'columns'):
        feature_names = X.columns.tolist()
else:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        


###############################################################################
# Load selected features - LOOP OVER ALL INCREMENTAL FEATURE FILES
###############################################################################


# ---- CONFIGURE THIS BLOCK ------------------------------------------------
# Path to folder containing selected_features_topXXX.csv files
feature_selection_dir = '../FEATURE_SELECTION/'

# Glob pattern – finds all selected_features_top<N>.csv files
feature_files = sorted(
    glob.glob(os.path.join(feature_selection_dir, 'selected_features_top*.csv')),
    key=lambda f: int(
        os.path.basename(f).replace('selected_features_top', '').replace('.csv', '')
    )
)
# --------------------------------------------------------------------------

if len(feature_files) == 0:
    raise FileNotFoundError(
        f"No selected_features_top*.csv files found in: {feature_selection_dir}"
    )

print(f"\nFound {len(feature_files)} feature file(s) to process:")
for f in feature_files:
    print(f"  {os.path.basename(f)}")

# Will accumulate one row per feature set for the final comparison
comparison_records = []

###############################################################################
# MAIN LOOP – one full hyperparameter tuning run per feature file
###############################################################################
for feature_file in feature_files:

    n_feat_label = (
        os.path.basename(feature_file)
        .replace('selected_features_top', '')
        .replace('.csv', '')
    )
    print(f"\n{'#'*80}")
    print(f"# PROCESSING: {os.path.basename(feature_file)}  (n_features = {n_feat_label})")
    print(f"{'#'*80}")

    # ------------------------------------------------------------------
    # Create dedicated output folder for this feature set
    # ------------------------------------------------------------------
    out_dir = os.path.join(os.getcwd(), f'features_top{n_feat_label}')
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output folder: {out_dir}")

    # ------------------------------------------------------------------
    # Load selected features and subset X
    # ------------------------------------------------------------------
    selected_features_df_loop = pd.read_csv(feature_file)
    selected_features = selected_features_df_loop['feature'].tolist()
    print(f"Loaded {len(selected_features)} features")

    if hasattr(X, 'columns'):
        X_selected = X[selected_features]
    else:
        selected_indices = [feature_names.index(feat) for feat in selected_features]
        X_selected = X[:, selected_indices]

    # ==================================================================
    # RANDOM FOREST NESTED CROSS-VALIDATION WITH HYPERPARAMETER TUNING
    # ==================================================================


    print("="*80)
    print("RANDOM FOREST NESTED CROSS-VALIDATION WITH HYPERPARAMETER TUNING")
    print("="*80)

    # Convert to arrays
    X_array = X_selected.values if hasattr(X_selected, 'values') else X_selected
    y_array = y.values if hasattr(y, 'values') else y

    # Get feature names
    if hasattr(X_selected, 'columns'):
        feature_names_selected = X_selected.columns.tolist()
    else:
        feature_names_selected = [f"Feature_{i}" for i in range(X_array.shape[1])]

    n_samples, n_features = X_array.shape
    n_classes = len(np.unique(y_array))

    print(f"\nDataset Information:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {n_classes}")
    print(f"  Class distribution: {dict(zip(*np.unique(y_array, return_counts=True)))}")

    # ------------------------------------------------------------------
    # HYPERPARAMETER SEARCH SPACE
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH SPACE")
    print("="*80)

    param_distributions = {
        'n_estimators': [100, 200, 300, 500, 1000],
        'max_depth': [10, 20, 30, 50, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
        'bootstrap': [True],
        'class_weight': ['balanced', 'balanced_subsample', None],
        'criterion': ['gini', 'entropy'],
        'max_leaf_nodes': [None, 50, 100, 200],
        'min_impurity_decrease': [0.0, 0.001, 0.01]
    }

    print("\nHyperparameters to tune:")
    for param, values in param_distributions.items():
        print(f"  {param:.<30} {values}")

    total_combinations = np.prod([len(v) for v in param_distributions.values()])
    print(f"\nTotal possible combinations: {total_combinations:,}")

    # ------------------------------------------------------------------
    # NESTED CROSS-VALIDATION SETUP
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("NESTED CROSS-VALIDATION SETUP")
    print("="*80)

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    n_iter_search = 10

    print(f"\nOuter CV folds: 5")
    print(f"Inner CV folds: 3")
    print(f"Random search iterations: {n_iter_search}")
    print(f"Total models to train: ~{5 * n_iter_search * 3:,}")

    # Storage
    outer_fold_results = []
    best_params_per_fold = []
    feature_importances_per_fold = []
    predictions_per_fold = []
    confusion_matrices = []
    all_metrics = defaultdict(list)

    # ------------------------------------------------------------------
    # NESTED CV LOOP
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("RUNNING NESTED CROSS-VALIDATION")
    print("="*80)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_array, y_array)):
        print(f"\n{'='*80}")
        print(f"OUTER FOLD {fold_idx + 1}/5")
        print(f"{'='*80}")

        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]

        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        print(f"Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"Test class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

        print(f"\nRunning hyperparameter search (inner CV)...")

        rf_base = RandomForestClassifier(
            random_state=42, n_jobs=8, oob_score=True, verbose=0
        )

        random_search = RandomizedSearchCV(
            estimator=rf_base,
            param_distributions=param_distributions,
            n_iter=n_iter_search,
            cv=inner_cv,
            scoring='balanced_accuracy',
            n_jobs=8,
            verbose=1,
            random_state=42 + fold_idx,
            return_train_score=True
        )

        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        best_cv_score = random_search.best_score_

        print(f"\n✓ Hyperparameter search complete!")
        print(f"  Best CV score (inner): {best_cv_score:.4f}")
        print(f"  Best parameters:")
        for param, value in best_params.items():
            print(f"    {param:.<30} {value}")

        best_params_per_fold.append(best_params)

        print(f"\nEvaluating best model on test set...")

        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)

        predictions_per_fold.append({
            'fold': fold_idx + 1,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'test_indices': test_idx
        })

        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)

        if n_classes == 2:
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted')
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')

        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        all_metrics['accuracy'].append(accuracy)
        all_metrics['balanced_accuracy'].append(balanced_acc)
        all_metrics['f1_score'].append(f1)
        all_metrics['precision'].append(precision)
        all_metrics['recall'].append(recall)
        all_metrics['roc_auc'].append(roc_auc)
        all_metrics['mcc'].append(mcc)
        all_metrics['kappa'].append(kappa)
        all_metrics['best_cv_score'].append(best_cv_score)

        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices.append(cm)
        feature_importances_per_fold.append(best_model.feature_importances_)

        print(f"\nFold {fold_idx + 1} Test Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  MCC: {mcc:.4f}")
        print(f"  Cohen's Kappa: {kappa:.4f}")

        outer_fold_results.append({
            'fold': fold_idx + 1,
            'best_params': best_params,
            'best_cv_score': best_cv_score,
            'test_accuracy': accuracy,
            'test_balanced_accuracy': balanced_acc,
            'test_f1': f1,
            'test_precision': precision,
            'test_recall': recall,
            'test_roc_auc': roc_auc,
            'test_mcc': mcc,
            'test_kappa': kappa,
            'confusion_matrix': cm
        })

    # ------------------------------------------------------------------
    # AGGREGATE RESULTS
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("NESTED CROSS-VALIDATION RESULTS SUMMARY")
    print("="*80)

    print("\nOverall Performance (Mean ± Std across 5 folds):")
    print("-" * 80)
    for metric_name, values in all_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {metric_name:.<30} {mean_val:.4f} ± {std_val:.4f}")

    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)

    # ------------------------------------------------------------------
    # TRAIN FINAL MODEL
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL ON FULL DATASET")
    print("="*80)

    best_fold_idx = np.argmax(all_metrics['balanced_accuracy'])
    final_params = best_params_per_fold[best_fold_idx]

    print(f"\nUsing hyperparameters from best fold (Fold {best_fold_idx + 1}):")
    for param, value in final_params.items():
        print(f"  {param:.<30} {value}")

    print("\nTraining final model on all data...")
    final_model = RandomForestClassifier(
        **final_params, random_state=42, n_jobs=8, oob_score=True, verbose=0
    )
    final_model.fit(X_array, y_array)

    print(f"✓ Final model trained!")
    print(f"  OOB Score: {final_model.oob_score_:.4f}")
    print(f"  Number of trees: {final_model.n_estimators}")
    print(f"  Number of features: {n_features}")

    # ------------------------------------------------------------------
    # EVALUATE FINAL MODEL (OOB)
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("FINAL MODEL EVALUATION (OOB)")
    print("="*80)

    y_final_pred_proba = final_model.oob_decision_function_
    y_final_pred_indices = np.argmax(y_final_pred_proba, axis=1)
    y_final_pred = final_model.classes_[y_final_pred_indices]
    valid_oob_mask = ~np.isnan(y_final_pred_proba).any(axis=1)

    print(f"\nOOB predictions available for {np.sum(valid_oob_mask)}/{len(y_array)} samples")

    y_array_valid = y_array[valid_oob_mask]
    y_final_pred_valid = y_final_pred[valid_oob_mask]
    y_final_pred_proba_valid = y_final_pred_proba[valid_oob_mask]

    y_true_str = np.array([label_mapping[val] for val in y_array_valid])
    y_pred_str = np.array([label_mapping[val] for val in y_final_pred_valid])

    print("\nFinal Model Classification Report (OOB):")
    print(classification_report(y_true_str, y_pred_str))

    final_accuracy = accuracy_score(y_true_str, y_pred_str)
    final_balanced_acc = balanced_accuracy_score(y_true_str, y_pred_str)
    final_f1 = f1_score(y_true_str, y_pred_str, average='weighted')
    final_precision = precision_score(y_true_str, y_pred_str, average='weighted', zero_division=0)
    final_recall = recall_score(y_true_str, y_pred_str, average='weighted')
    final_mcc = matthews_corrcoef(y_true_str, y_pred_str)
    final_kappa = cohen_kappa_score(y_true_str, y_pred_str)

    y_true_bin_final = label_binarize(y_array_valid, classes=sorted(np.unique(y_array_valid)))
    final_roc_auc = roc_auc_score(
        y_true_bin_final, y_final_pred_proba_valid, multi_class='ovr', average='weighted'
    )

    print("\nFinal Model OOB Metrics Summary:")
    print("-" * 80)
    print(f"  Accuracy:            {final_accuracy:.4f}")
    print(f"  Balanced Accuracy:   {final_balanced_acc:.4f}")
    print(f"  F1 Score (weighted): {final_f1:.4f}")
    print(f"  Precision (weighted):{final_precision:.4f}")
    print(f"  Recall (weighted):   {final_recall:.4f}")
    print(f"  ROC AUC (weighted):  {final_roc_auc:.4f}")
    print(f"  MCC:                 {final_mcc:.4f}")
    print(f"  Cohen's Kappa:       {final_kappa:.4f}")

    print("\nFinal Model Confusion Matrix (OOB):")
    cm_final = confusion_matrix(y_true_str, y_pred_str)
    unique_labels_oob = sorted(np.unique(y_true_str))
    cm_final_df = pd.DataFrame(cm_final, index=unique_labels_oob, columns=unique_labels_oob)
    print(cm_final_df)

    print("\nPer-Class Performance (OOB):")
    print("-" * 80)
    for label in unique_labels_oob:
        mask = (y_true_str == label)
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_str[mask] == y_true_str[mask])
            print(f"  {label:.<20} Accuracy: {class_acc:.4f} ({np.sum(mask)} samples)")

    # ------------------------------------------------------------------
    # SAVE MODELS AND RESULTS  →  all paths redirected to out_dir
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("SAVING MODELS AND RESULTS")
    print("="*80)

    model_filename = os.path.join(out_dir, "random_forest_final_model.pkl")
    joblib.dump(final_model, model_filename)
    print(f"✓ Saved final model: {model_filename}")

    model_metadata = {
        'model': final_model,
        'feature_names': feature_names_selected,
        'hyperparameters': final_params,
        'n_features': n_features,
        'n_classes': n_classes,
        'cv_results': outer_fold_results,
        'performance_metrics': all_metrics,
        'class_names': np.unique(y_array).tolist(),
        'label_mapping': label_mapping,
        'final_model_oob_metrics': {
            'accuracy': final_accuracy,
            'balanced_accuracy': final_balanced_acc,
            'f1_score': final_f1,
            'precision': final_precision,
            'recall': final_recall,
            'roc_auc': final_roc_auc,
            'mcc': final_mcc,
            'kappa': final_kappa
        }
    }
    metadata_filename = os.path.join(out_dir, "random_forest_model_with_metadata.pkl")
    joblib.dump(model_metadata, metadata_filename)
    print(f"✓ Saved model with metadata: {metadata_filename}")

    results_df = pd.DataFrame(outer_fold_results)
    results_df.to_csv(os.path.join(out_dir, "nested_cv_results_per_fold.csv"), index=False)
    print(f"✓ Saved CV results: nested_cv_results_per_fold.csv")

    metrics_summary = pd.DataFrame({
        'Metric': list(all_metrics.keys()),
        'Mean': [np.mean(v) for v in all_metrics.values()],
        'Std': [np.std(v) for v in all_metrics.values()],
        'Min': [np.min(v) for v in all_metrics.values()],
        'Max': [np.max(v) for v in all_metrics.values()]
    })
    metrics_summary.to_csv(os.path.join(out_dir, "performance_metrics_summary.csv"), index=False)
    print(f"✓ Saved metrics summary: performance_metrics_summary.csv")

    avg_feature_importance = np.mean(feature_importances_per_fold, axis=0)
    std_feature_importance = np.std(feature_importances_per_fold, axis=0)
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names_selected,
        'Importance_Mean': avg_feature_importance,
        'Importance_Std': std_feature_importance,
        'Importance_CV': std_feature_importance / (avg_feature_importance + 1e-10),
        'Final_Model_Importance': final_model.feature_importances_
    })
    feature_importance_df = feature_importance_df.sort_values('Importance_Mean', ascending=False)
    feature_importance_df.to_csv(os.path.join(out_dir, "feature_importances.csv"), index=False)
    print(f"✓ Saved feature importances: feature_importances.csv")

    params_df = pd.DataFrame(best_params_per_fold)
    params_df.insert(0, 'Fold', range(1, 6))
    params_df.to_csv(os.path.join(out_dir, "best_hyperparameters_per_fold.csv"), index=False)
    print(f"✓ Saved hyperparameters: best_hyperparameters_per_fold.csv")

    with open(os.path.join(out_dir, 'model_training_summary.txt'), 'w') as f:
        f.write("="*80 + "\n")
        f.write("RANDOM FOREST MODEL TRAINING SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write("DATASET INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total samples:        {n_samples}\n")
        f.write(f"Total features:       {n_features}\n")
        f.write(f"Number of classes:    {n_classes}\n")
        f.write(f"Class distribution:   {dict(zip(*np.unique(y_array, return_counts=True)))}\n\n")
        f.write("NESTED CROSS-VALIDATION RESULTS (5-Fold)\n")
        f.write("-" * 80 + "\n")
        for metric_name, values in all_metrics.items():
            f.write(f"{metric_name:.<30} {np.mean(values):.4f} ± {np.std(values):.4f}\n")
        f.write("\n")
        f.write("FINAL MODEL HYPERPARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Selected from Fold {best_fold_idx + 1} (best balanced accuracy)\n\n")
        for param, value in final_params.items():
            f.write(f"{param:.<30} {value}\n")
        f.write("\n")
        f.write("FINAL MODEL OOB PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"OOB Score (sklearn):     {final_model.oob_score_:.4f}\n")
        f.write(f"OOB Samples evaluated:   {np.sum(valid_oob_mask)}/{len(y_array)}\n")
        f.write(f"Accuracy:                {final_accuracy:.4f}\n")
        f.write(f"Balanced Accuracy:       {final_balanced_acc:.4f}\n")
        f.write(f"F1 Score (weighted):     {final_f1:.4f}\n")
        f.write(f"Precision (weighted):    {final_precision:.4f}\n")
        f.write(f"Recall (weighted):       {final_recall:.4f}\n")
        f.write(f"ROC AUC (weighted):      {final_roc_auc:.4f}\n")
        f.write(f"MCC:                     {final_mcc:.4f}\n")
        f.write(f"Cohen's Kappa:           {final_kappa:.4f}\n\n")
        f.write("FINAL MODEL CLASSIFICATION REPORT (OOB)\n")
        f.write("-" * 80 + "\n")
        f.write(classification_report(y_true_str, y_pred_str))
        f.write("\n")
        f.write("FINAL MODEL CONFUSION MATRIX (OOB)\n")
        f.write("-" * 80 + "\n")
        f.write(str(cm_final_df))
        f.write("\n\n")
        f.write("PER-CLASS PERFORMANCE (OOB)\n")
        f.write("-" * 80 + "\n")
        for label in unique_labels_oob:
            mask = (y_true_str == label)
            if np.sum(mask) > 0:
                class_acc = np.mean(y_pred_str[mask] == y_true_str[mask])
                f.write(f"{label:.<20} Accuracy: {class_acc:.4f} ({np.sum(mask)} samples)\n")
        f.write("\n")
        f.write("HYPERPARAMETERS ACROSS ALL FOLDS\n")
        f.write("-" * 80 + "\n")
        f.write(str(params_df))
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("="*80 + "\n")
    print(f"✓ Saved comprehensive summary: model_training_summary.txt")

    # ------------------------------------------------------------------
    # VISUALIZATIONS  →  saved to out_dir
    # ------------------------------------------------------------------
    unique_string_labels = sorted([label_mapping[num] for num in np.unique(y_array)])

    ###########################################################################
    # Figure 1: Main 4-panel analysis plot
    ###########################################################################
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    metrics_to_plot = ['balanced_accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']

    ax3 = fig.add_subplot(gs[0, 0])
    metrics_data = [all_metrics[m] for m in metrics_to_plot]
    bp = ax3.boxplot(metrics_data,
                     labels=[m.replace('_', '\n').title() for m in metrics_to_plot],
                     patch_artist=True, showmeans=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('a) Distribution of Metrics Across CV Folds', fontsize=12, fontweight='normal')
    ax3.grid(True, alpha=0.3, axis='y')

    ax4 = fig.add_subplot(gs[0, 1])
    top_20_features = feature_importance_df.head(20)
    y_pos = np.arange(len(top_20_features))
    ax4.barh(y_pos, top_20_features['Importance_Mean'],
             xerr=top_20_features['Importance_Std'], capsize=8)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(top_20_features['Feature'], fontsize=8)
    ax4.set_xlabel('Mean Importance (±Std)', fontsize=8)
    ax4.set_title('b) Top 20 Feature Importances', fontsize=12, fontweight='normal')
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis='x')

    ax10 = fig.add_subplot(gs[1, 0])
    if n_classes == 2:
        y_true_binary = y_array_valid
        y_proba_oob = y_final_pred_proba_valid[:, 1]
        fpr, tpr, _ = roc_curve(y_true_binary, y_proba_oob)
        roc_auc_val = auc(fpr, tpr)
        ax10.plot(fpr, tpr, color='darkblue', lw=3,
                  label=f'Final Model OOB (AUC={roc_auc_val:.3f})')
        ax10.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax10.set_xlabel('False Positive Rate', fontsize=8)
        ax10.set_ylabel('True Positive Rate', fontsize=8)
        ax10.set_title(f'c) Final Model ROC Curve (OOB, n={np.sum(valid_oob_mask)})',
                       fontsize=12, fontweight='normal')
        ax10.legend(fontsize=8)
        ax10.grid(True, alpha=0.3)
    else:
        y_true_bin = label_binarize(y_array_valid, classes=sorted(np.unique(y_array_valid)))
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
        for i, numeric_label in enumerate(sorted(np.unique(y_array))):
            string_label = label_mapping[numeric_label]
            if numeric_label in y_array_valid:
                class_idx = list(sorted(np.unique(y_array))).index(numeric_label)
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_final_pred_proba_valid[:, class_idx])
                roc_auc_val = auc(fpr, tpr)
                ax10.plot(fpr, tpr, color=colors[i], lw=3,
                          label=f'{string_label} (AUC={roc_auc_val:.3f})')
        ax10.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax10.set_xlabel('False Positive Rate', fontsize=8)
        ax10.set_ylabel('True Positive Rate', fontsize=8)
        ax10.set_title(f'c) Final Model ROC Curves (OOB, One-vs-Rest, n={np.sum(valid_oob_mask)})',
                       fontsize=12, fontweight='normal')
        ax10.legend(fontsize=8, loc='lower right')
        ax10.grid(True, alpha=0.3)

    ax11 = fig.add_subplot(gs[1, 1])
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Greens', ax=ax11,
                xticklabels=unique_labels_oob, yticklabels=unique_labels_oob,
                cbar_kws={'label': 'Count'})
    ax11.set_xlabel('Predicted Label', fontsize=8)
    ax11.set_ylabel('True Label', fontsize=8)
    ax11.set_yticklabels(ax11.get_yticklabels(), rotation=45, va='center')
    ax11.set_title(f'd) Final Model Confusion Matrix (OOB, n={np.sum(valid_oob_mask)})',
                   fontsize=12, fontweight='normal')

    plt.savefig(os.path.join(out_dir, "random_forest_FINAL.svg"),
                format="svg", bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: random_forest_FINAL.svg")

    ###########################################################################
    # Figure 2: Short ROC + confusion matrix
    ###########################################################################
    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

    ax10 = fig.add_subplot(gs[0, 0])
    if n_classes == 2:
        y_true_binary = y_array_valid
        y_proba_oob = y_final_pred_proba_valid[:, 1]
        fpr, tpr, _ = roc_curve(y_true_binary, y_proba_oob)
        roc_auc_val = auc(fpr, tpr)
        ax10.plot(fpr, tpr, color='darkblue', lw=3,
                  label=f'Final Model OOB (AUC={roc_auc_val:.3f})')
        ax10.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax10.set_xlabel('False Positive Rate', fontsize=8)
        ax10.set_ylabel('True Positive Rate', fontsize=8)
        ax10.set_title(f'a) Final Model ROC Curve', fontsize=12, fontweight='normal')
        ax10.legend(fontsize=8)
        ax10.grid(True, alpha=0.3)
    else:
        y_true_bin = label_binarize(y_array_valid, classes=sorted(np.unique(y_array_valid)))
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
        for i, numeric_label in enumerate(sorted(np.unique(y_array))):
            string_label = label_mapping[numeric_label]
            if numeric_label in y_array_valid:
                class_idx = list(sorted(np.unique(y_array))).index(numeric_label)
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_final_pred_proba_valid[:, class_idx])
                roc_auc_val = auc(fpr, tpr)
                ax10.plot(fpr, tpr, color=colors[i], lw=3,
                          label=f'{string_label} (AUC={roc_auc_val:.3f})')
        ax10.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax10.set_xlabel('False Positive Rate', fontsize=8)
        ax10.set_ylabel('True Positive Rate', fontsize=8)
        ax10.set_title(f'a) Final Model ROC Curves', fontsize=12, fontweight='normal')
        ax10.legend(fontsize=8, loc='lower right')
        ax10.grid(True, alpha=0.3)

    ax11 = fig.add_subplot(gs[0, 1])
    cm_final2 = confusion_matrix(y_true_str, y_pred_str)
    unique_labels_oob2 = sorted(np.unique(y_true_str))
    sns.heatmap(cm_final2, annot=True, fmt='d', cmap='Greens', ax=ax11,
                xticklabels=unique_labels_oob2, yticklabels=unique_labels_oob2,
                cbar_kws={'label': 'Count'})
    ax11.set_xlabel('Predicted Label', fontsize=8)
    ax11.set_ylabel('True Label', fontsize=8)
    ax11.set_yticklabels(ax11.get_yticklabels(), rotation=45, va='center')
    ax11.set_title(f'b) Final Model Confusion Matrix', fontsize=12, fontweight='normal')

    plt.savefig(os.path.join(out_dir, "random_forest_FINAL_short.svg"),
                format="svg", bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: random_forest_FINAL_short.svg")

    # ------------------------------------------------------------------
    # FINAL SUMMARY (printed, not saved separately)
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\n📊 Dataset:")
    print(f"   • Total samples: {n_samples}")
    print(f"   • Total features: {n_features}")
    print(f"   • Number of classes: {n_classes}")
    print(f"   • Class labels: {unique_string_labels}")
    print(f"\n🎯 Nested Cross-Validation Performance:")
    print(f"   • Balanced Accuracy: {np.mean(all_metrics['balanced_accuracy']):.4f} ± {np.std(all_metrics['balanced_accuracy']):.4f}")
    print(f"   • F1 Score: {np.mean(all_metrics['f1_score']):.4f} ± {np.std(all_metrics['f1_score']):.4f}")
    print(f"   • ROC AUC: {np.mean(all_metrics['roc_auc']):.4f} ± {np.std(all_metrics['roc_auc']):.4f}")
    print(f"   • MCC: {np.mean(all_metrics['mcc']):.4f} ± {np.std(all_metrics['mcc']):.4f}")
    print(f"\n🌲 Final Model:")
    print(f"   • OOB Score: {final_model.oob_score_:.4f}")
    print(f"   • n_estimators: {final_model.n_estimators}")
    print(f"   • max_depth: {final_model.max_depth}")
    print(f"   • max_features: {final_model.max_features}")

    # ------------------------------------------------------------------
    # Accumulate row for cross-feature-set comparison
    # ------------------------------------------------------------------
    comparison_records.append({
        'n_features': int(n_feat_label),
        'feature_file': os.path.basename(feature_file),
        # CV means (nested cross-validation)
        'cv_accuracy_mean':          np.mean(all_metrics['accuracy']),
        'cv_accuracy_std':           np.std(all_metrics['accuracy']),
        'cv_balanced_accuracy_mean': np.mean(all_metrics['balanced_accuracy']),
        'cv_balanced_accuracy_std':  np.std(all_metrics['balanced_accuracy']),
        'cv_f1_mean':                np.mean(all_metrics['f1_score']),
        'cv_f1_std':                 np.std(all_metrics['f1_score']),
        'cv_precision_mean':         np.mean(all_metrics['precision']),
        'cv_precision_std':          np.std(all_metrics['precision']),
        'cv_recall_mean':            np.mean(all_metrics['recall']),
        'cv_recall_std':             np.std(all_metrics['recall']),
        'cv_roc_auc_mean':           np.mean(all_metrics['roc_auc']),
        'cv_roc_auc_std':            np.std(all_metrics['roc_auc']),
        'cv_mcc_mean':               np.mean(all_metrics['mcc']),
        'cv_mcc_std':                np.std(all_metrics['mcc']),
        # Final model OOB metrics
        'oob_accuracy':          final_accuracy,
        'oob_balanced_accuracy': final_balanced_acc,
        'oob_f1':                final_f1,
        'oob_precision':         final_precision,
        'oob_recall':            final_recall,
        'oob_roc_auc':           final_roc_auc,
        'oob_mcc':               final_mcc,
        'oob_kappa':             final_kappa,
        'oob_score_sklearn':     final_model.oob_score_,
    })

    print(f"\n✓ Finished feature set: top{n_feat_label}")
    print(f"  Results saved to: {out_dir}\n")

    # Close all figures to free memory before next iteration
    plt.close('all')

# END OF MAIN LOOP
###############################################################################

###############################################################################
# CROSS-FEATURE-SET COMPARISON
###############################################################################

print("\n" + "="*80)
print("CROSS-FEATURE-SET COMPARISON")
print("="*80)

comparison_df = pd.DataFrame(comparison_records).sort_values('n_features')
print("\n" + comparison_df.to_string(index=False))

# Save comparison table
comparison_df.to_csv("feature_set_comparison_metrics.csv", index=False)
print("\n✓ Saved: feature_set_comparison_metrics.csv")

###############################################################################
# COMPARISON PLOT – performance vs number of features
###############################################################################

metrics_to_compare = [
    ('cv_f1_mean',                'cv_f1_std',                'F1 Score (weighted)'),
    ('cv_accuracy_mean',          'cv_accuracy_std',          'Accuracy'),
    ('cv_precision_mean',         'cv_precision_std',         'Precision'),
    ('cv_recall_mean',            'cv_recall_std',            'Recall'),
    ('cv_roc_auc_mean',           'cv_roc_auc_std',           'ROC AUC'),
    ('cv_balanced_accuracy_mean', 'cv_balanced_accuracy_std', 'Balanced Accuracy'),
]

n_metrics = len(metrics_to_compare)
ncols = 3
nrows = (n_metrics + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
axes = axes.flatten()

x = comparison_df['n_features'].values

for ax_idx, (mean_col, std_col, title) in enumerate(metrics_to_compare):
    ax = axes[ax_idx]
    means = comparison_df[mean_col].values
    stds  = comparison_df[std_col].values

    ax.plot(x, means, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.fill_between(x, means - stds, means + stds, alpha=0.25, color='steelblue',
                    label='±1 SD')

    # Mark the best value
    best_idx = np.argmax(means)
    ax.scatter(x[best_idx], means[best_idx], color='red', zorder=5, s=120,
               label=f'Best: {x[best_idx]:,} features\n({means[best_idx]:.4f})')

    ax.set_xlabel('Number of Features', fontsize=11)
    ax.set_ylabel(title, fontsize=11)
    ax.set_title(f'{title} vs Number of Features', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{v:,}' for v in x], rotation=45, ha='right', fontsize=9)

# Hide any unused subplot axes
for ax_idx in range(n_metrics, len(axes)):
    axes[ax_idx].set_visible(False)

plt.suptitle('Model Performance vs Number of Selected Features\n(Nested CV, mean ± 1 SD)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("feature_set_comparison_plot.svg", format="svg", bbox_inches='tight')
plt.savefig("feature_set_comparison_plot.png", dpi=200, bbox_inches='tight')
plt.show()
print("✓ Saved: feature_set_comparison_plot.svg / .png")

###############################################################################
# FINAL BEST FEATURE SET RECOMMENDATION
###############################################################################

print("\n" + "="*80)
print("BEST FEATURE SET RECOMMENDATION")
print("="*80)

for mean_col, _, title in metrics_to_compare:
    best_row = comparison_df.loc[comparison_df[mean_col].idxmax()]
    print(f"  Best {title:.<35} {int(best_row['n_features']):>6,} features  "
          f"({best_row[mean_col]:.4f})")

print("\n" + "="*80)
print("✓ ALL FEATURE SETS PROCESSED!")
print("="*80)