rm(list = ls())  # vymazu Envrironment
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# BiocManager::install("zwdzwd/sesame")

# processing
suppressPackageStartupMessages(library(sesame))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(openxlsx))
suppressPackageStartupMessages(library(dplyr))
sesameDataCache()

sink(file="sessionInfo.txt")
sessionInfo()
sink()

################################################################################
# SAMPLESHEETS
SAMPLESv1 <- openxlsx::read.xlsx("../sample_sheet_Sept2024_EPICv1.xlsx",
                               sheet = 1)
setDT(SAMPLESv1)
SAMPLESv2 <- openxlsx::read.xlsx("../sample_sheet_Sept2024_EPICv2.xlsx",
                               sheet = 1)
setDT(SAMPLESv2)
SAMPLES <- rbindlist(list(SAMPLESv1,SAMPLESv2))

################################################################################
## ANOTACE
ANOTACE <- fread("../sample_sheet_ANOTACE_January2026_Correct_Shivani_CODES_TISSUEorigin.tsv")
ANOTACE <- unique(ANOTACE, by="Sample_Name")
ANOTACE$ORIGIN <- "CEITEC"

SAMPLES[, Sample_Name := gsub("[[:space:]]", "", Sample_Name)] #remove whitespace
ANOTACE[, Sample_Name := gsub("[[:space:]]", "", Sample_Name)] #remove whitespace

# merge
SAMPLES <- merge(SAMPLES,ANOTACE, by="Sample_Name",all.x=TRUE,all.y=TRUE)


############################
# DELETE THIS:
SAMPLES <- SAMPLES[!Basename=="207714950054_R01C01",] #pediatric Sample
SAMPLES <- SAMPLES[!Basename=="207714950077_R08C01",] #LOST IDAT
# DELETE THIS:
# BAD BETA VALUES DISTRIB. / BAD QC
SAMPLES <- SAMPLES[!Sample_Name=="FNM_1",]
SAMPLES <- SAMPLES[!Sample_Name=="FNM_20",]
SAMPLES <- SAMPLES[!Sample_Name=="NU210010",]
SAMPLES <- SAMPLES[!Sample_Name=="NU210065",]
SAMPLES <- SAMPLES[!Sample_Name=="NU210065",]
SAMPLES <- SAMPLES[!Sample_Name=="NU210065",]

################################################################################
# GEO ANOTACE & SAMPLESHEET
ANOTACE_GEO <- fread("../GEO_sample_sheet_CURATED_January2026.tsv")

ANOTACE_GEO$ORIGIN <- "GEO"

############################
# ANOTACE_GEO <- ANOTACE_GEO[!Sentrix_ID=="203273200108",] # obsahuje 2 vzorky Coefficients not estimable: Sentrix_ID203273200108 
# ANOTACE_GEO <- ANOTACE_GEO[!Sentrix_ID=="203259190053",] # obsahuje 2 vzorky FCD3B Coefficients not estimable: Sentrix_ID203259190053 



ANOTACE_GEO <- ANOTACE_GEO %>%
  group_by(SUBTYPE) %>%
  mutate(NewCODE = paste0(SUBTYPE, "_GEO_", row_number())) %>%
  ungroup()

#append
SAMPLES <- rbindlist(list(SAMPLES,ANOTACE_GEO),fill=TRUE)

rm(SAMPLESv1,SAMPLESv2,ANOTACE,ANOTACE_GEO)

################################################################################
# CHIP VERSION
SAMPLES$CHIP <- "V1"
SAMPLES[SAMPLES$Sentrix_ID=="207690200072","CHIP"] <- "V2" 
SAMPLES[SAMPLES$Sentrix_ID=="208553430088","CHIP"] <- "V2" 
SAMPLES[SAMPLES$Sentrix_ID=="208553430049","CHIP"] <- "V2" 
SAMPLES[SAMPLES$Sentrix_ID=="208553430054","CHIP"] <- "V2" 
SAMPLES[SAMPLES$Sentrix_ID=="208553430058","CHIP"] <- "V2"
SAMPLES$CHIP <- as.factor(SAMPLES$CHIP)


################################################################################
# remove those without TYPE
SAMPLES <- SAMPLES[!Sentrix_ID=="",]
SAMPLES <- SAMPLES[!TYPE=="",]
SAMPLES <- SAMPLES[!(diagnosis_status == "skip") | is.na(diagnosis_status)]

table(SAMPLES$TYPE)
table(SAMPLES$SUBTYPE)

# remove unwanted TYPES
# SAMPLES <- SAMPLES[!TYPE=="MOGHE",]
# SAMPLES <- SAMPLES[!TYPE=="HS",]
# SAMPLES <- SAMPLES[!TYPE=="complexMCD",]
# SAMPLES <- SAMPLES[!SUBTYPE=="FCD1B",]
# 
# SAMPLES <- SAMPLES[!TYPE=="HME",]
# SAMPLES <- SAMPLES[!TYPE=="mMCD",]
# SAMPLES <- SAMPLES[!TYPE=="PMG",]
# SAMPLES <- SAMPLES[!TYPE=="TLE",]
# SAMPLES <- SAMPLES[!TYPE=="TSC",]

# #replace CTRL with PM as SUBTYPE
SAMPLES[TYPE=="CTRL","TYPE"] <- "PM"
SAMPLES[TYPE=="PM","SUBTYPE"] <- "PM"



table(SAMPLES$TYPE)
table(SAMPLES$SUBTYPE)

SAMPLES <- unique(SAMPLES, by="Sample_Name")
SAMPLES$AGE <- as.integer(SAMPLES$AGE)
SAMPLES$SEX <- as.factor(SAMPLES$SEX)
SAMPLES$TYPE <- as.factor(SAMPLES$TYPE)
SAMPLES$SUBTYPE <- as.factor(SAMPLES$SUBTYPE)

names(SAMPLES)
SAMPLES[,c("longRNAseq_RUN","smallRNAseq_RUN","diagnosis_status","INFO","V16", "Sample_Plate","Sample_Group","Pool_ID","Project","Sample_Well" ) := NULL]


openxlsx::write.xlsx(SAMPLES,file="SAMPLES_MLcohort_used.xlsx")

################################################################################
# openSesame returns beta values
################################################################################
# EPICv1 +GEO
EPICV1path <- "/home/rj/4TB/PORTABLE_DATA/CIPY_BRAZDIL_FCD/ALL_IDATS_TOGETHER_ML_COHORT"
IDATprefixesEPICv1 <- sesame::searchIDATprefixes(EPICV1path)

EPICV1beta <- sesame::openSesame(IDATprefixesEPICv1,
                             # collapseToPfx = TRUE, collapseMethod = "mean", 
                             platform="EPIC",
                             prep="QCDPB", #"QCDPB", default=Best-practice pipeline: noob + pOOBAH + masking, etc. ✅
                             BPPARAM=BiocParallel::MulticoreParam(8))


save(EPICV1beta, SAMPLES, file="/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort/FCD_sesame__EPICV1_February2026.Rdata", compress=TRUE)

################################################################################
# EPICv2
EPICV2path <- "/home/rj/4TB/PORTABLE_DATA/CIPY_BRAZDIL_FCD/ALL_IDATS_TOGETHER_EPICv2"
IDATprefixesEPICv2 <- sesame::searchIDATprefixes(EPICV2path)

EPICV2beta <- sesame::openSesame(IDATprefixesEPICv2, 
                             collapseToPfx = TRUE, collapseMethod = "mean",
                             platform="EPICv2",
                             prep="QCDPB", #"QCDPB", default=Best-practice pipeline: noob + pOOBAH + masking, etc. ✅
                             BPPARAM=BiocParallel::MulticoreParam(8))


save(EPICV2beta, SAMPLES, file="/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort/FCD_sesame__EPICV2_February2026.Rdata", compress=TRUE)



# https://www.bioconductor.org/packages/release/bioc/vignettes/sesame/inst/doc/sesame.html#Data_Preprocessing
# QCDPB = 
# Q	qualityMask	Mask probes of poor design
# C	inferInfiniumIChannel	Infer channel for Infinium-I probes
# D	dyeBiasNL	Dye bias correction (non-linear)
# P	pOOBAH	Detection p-value masking using oob
# B	noob	Background subtraction using oob


# metadata frame for sample information, column names are predictor variables (e.g.,
# sex, age, treatment, tumor/normal etc) and are referenced in formula. Rows are
# samples. When the betas argument is a SummarizedExperiment object, this is
# ignored. colData(betas) will be used instead. The row order of the data frame
# must match the column order of the beta value matrix.

################################################################################
# Project probe IDs
# cg_v1 = names(sesameData_getManifestGRanges("EPIC"))
## only mappable probes, return mapping from MSA to HM450
# head(mLiftOver(cg_v1, "EPICv2"))


# cg_v2 = names(sesameData_getManifestGRanges("EPICv2"))
# cg_v2 = grep("cg", cg_v2, value=TRUE)
## only mappable probes, return mapping from HM450 to EPICv2
# head(mLiftOver(cg_v2, "EPICv1"))
################################################################################
# MERGE BETA VALUES 
EPICV1beta <- as.data.frame(EPICV1beta)
EPICV1beta$IlmnID <- rownames(EPICV1beta)
EPICV2beta <- as.data.frame(EPICV2beta)
EPICV2beta$IlmnID <- rownames(EPICV2beta)

EPICbeta <- merge(EPICV1beta, EPICV2beta, by="IlmnID")
dim(EPICbeta)

IlmnID <- EPICbeta$IlmnID

EPICbeta$IlmnID <- NULL
rownames(EPICbeta) <- IlmnID


################################################################################
### IMPUTE
library(impute)

# Step 1: Remove probes with too many NA
probe_missing <- rowMeans(is.na(EPICbeta))
keep <- probe_missing <= 0.05
EPICbeta_keep <- EPICbeta[keep, ]

cat(sum(!keep), "probes removed due to high missingness\n")

# Step 2: Check remaining missingness
if (anyNA(EPICbeta_keep)) {
  EPICbeta_imputed <- impute.knn(as.matrix(EPICbeta_keep))$data
  cat("kNN imputation performed on remaining probes.\n")
} else {
  EPICbeta_imputed <- EPICbeta_keep
  cat("No missing values remaining.\n")
}

### PLOTS
# Distribution of values (before vs after)
# Flatten values
vals_before <- as.vector(as.matrix(na.omit(EPICbeta_keep)))
vals_after <- as.vector(EPICbeta_imputed)


options(scipen=999)
# Open SVG device
svg("imputation_comparison.svg", width = 10, height = 5)

# Split plotting area into 1 row, 2 columns
par(mfrow = c(1, 2))

# BEFORE
hist(vals_before,
     breaks = 100,
     main = "Distribution BEFORE Imputation",
     xlab = "Beta / M values",
     col = "grey",
     border = "white")

# AFTER
hist(vals_after,
     breaks = 100,
     main = "Distribution AFTER kNN Imputation",
     xlab = "Beta / M values",
     col = "grey",
     border = "white")

# Close device
dev.off()
Sys.sleep(2)

# Missingness per probe (very informative)
missing_per_probe <- rowMeans(is.na(EPICbeta_keep))

hist(missing_per_probe,
     breaks = 50,
     main = "Proportion of Missing Values per Probe",
     xlab = "Fraction Missing")


sum(is.na(EPICbeta_imputed))


dim(EPICbeta_imputed)
length(which(probe_missing<=0.05))
# 10% [1] ~313 228    468
# 5% [1] 233 968    468
# 1% [1] 109 053    468


EPICbeta <- as.data.frame(EPICbeta_imputed)
IlmnID <- rownames(EPICbeta)

################################################################################
# GET M VALUES 
EPICM <- log2(EPICbeta/(1-EPICbeta))
rownames(EPICM) <- IlmnID

################################################################################
# crop by SAMPLES
EPICbeta <- EPICbeta[,colnames(EPICbeta) %in% SAMPLES$Basename]
EPICM <- EPICM[,colnames(EPICM) %in% SAMPLES$Basename]

################################################################################
# Rename by newCOde
colnames(EPICbeta) <- unlist(SAMPLES[match(colnames(EPICbeta), SAMPLES$Basename), "NewCODE"])
colnames(EPICM) <- unlist(SAMPLES[match(colnames(EPICM), SAMPLES$Basename), "NewCODE"])

# sort by SAMPLES - is different for DT and DF
SAMPLES <- SAMPLES[match(colnames(EPICbeta), SAMPLES$NewCODE), ]
SAMPLES <- SAMPLES[match(colnames(EPICM), SAMPLES$NewCODE), ]

# check if
all(colnames(EPICM)==SAMPLES$NewCODE)
all(colnames(EPICbeta)==SAMPLES$NewCODE)

# check if
all(rownames(EPICM)==IlmnID)
all(rownames(EPICbeta)==IlmnID)

# remove NAs
# EPICbeta <- na.omit(EPICbeta)
# EPICM <- na.omit(EPICM)





################################################################################
# LOAD IDATS

save(EPICbeta, EPICM, SAMPLES, file="/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort_February2026_imputed/FCD_sesame_February2026_imputed.Rdata", compress=TRUE)

