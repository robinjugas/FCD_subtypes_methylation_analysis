# FCD_subtypes_methylation_analysis

Scripts and resources for the manuscript:

**_Methylation-Based Classification of Focal Cortical Dysplasia Subtypes Using Random Forest_**  
(*manuscript in preparation; not yet published*)

> **Repository status:** the current folder structure is **not the final arrangement** of the repo. Files and directories may be renamed, merged, or moved as the workflow is finalized and prepared for release.

---

## Contents (current layout)

- **`R_sesame_internalCohort_preparation/`**  
  R scripts to preprocess the internal cohort from IDATs using `sesame` (loading IDATs, probe filtering, and batch-effect correction).
  - `1_loadIDATs_SESAME_EPICv1+v2.R`
  - `2_filter_probes.R`
  - `3_removeBatchEffect_SUBTYPE_v2.R`

- **`R_sesame_ClassifierCohort_preparation/`**  
  R scripts to preprocess / harmonize the classifier cohort (load IDATs + imputation, probe filtering, batch-effect correction).
  - `1_loadIDATs_impute_SESAME_BRAZDIL+KOBOW.R`
  - `2_filter_probes.R`
  - `3_removeBatchEffect_FINAL_DECIDED_transposed_noUsed.R`

- **`SCIKIT_ClassifierCohort_RandomForest/`**  
  Files related to Random Forest model development (feature selection, hyperparameter tuning, and validation), plus an R helper script.
  - `FEATURE_SELECTION/`
  - `HYPERPARAMETER_TUNING/`
  - `MODEL_VALIDATION/`
  - `FEATURES_ANALYZED/`
  - `separate_train_validation.R`

- **`IlluminaEPIC_probes2filter/`**  
  Probe filtering resources for Illumina 450k and EPIC/850k

