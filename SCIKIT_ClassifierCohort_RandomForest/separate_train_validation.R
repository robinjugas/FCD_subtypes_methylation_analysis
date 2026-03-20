# Load packages
rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

suppressPackageStartupMessages(library(data.table))



SAMPLES <- fread("/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort_February2026_imputed/FCD_EPIC_SAMPLES_February2026_forPython.tsv", sep = "\t")

SAMPLES$SUBTYPE <- as.factor(SAMPLES$SUBTYPE)
SAMPLES$SUBTYPE_NUM <- as.numeric(SAMPLES$SUBTYPE)

fwrite(SAMPLES, file = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort_February2026_imputed/FCD_EPIC_SAMPLES_February2026_forPython_SUBTYPENUM.tsv", sep = "\t")

################################################################################
# TRAIN AND VALIDATION COHORT
table(SAMPLES$SUBTYPE)

set.seed(123)

idx <- caret::createDataPartition(SAMPLES$SUBTYPE, p = 0.1, list = FALSE)

validation <- SAMPLES[idx, ]
train <- SAMPLES[-idx, ]

table(validation$SUBTYPE)
table(train$SUBTYPE)



################################################################################


fwrite(train, file = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort_February2026_imputed/FCD_EPIC_SAMPLES_February2026_forPython_TRAIN.tsv", sep = "\t")
fwrite(validation, file = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort_February2026_imputed/FCD_EPIC_SAMPLES_February2026_forPython_VALIDATION.tsv", sep = "\t")
