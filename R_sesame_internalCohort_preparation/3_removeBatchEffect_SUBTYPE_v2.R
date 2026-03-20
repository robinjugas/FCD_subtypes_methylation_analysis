rm(list = ls())  # vymazu Envrironment
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# processing
suppressPackageStartupMessages(library(minfi))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(limma))
suppressPackageStartupMessages(library(sva))


################################################################################
load("/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_met/FCD_sesame_February2026_filteredprobes.Rdata")

SAMPLES$Sentrix_Position <- factor(SAMPLES$Sentrix_Position)
SAMPLES$Sentrix_ID <- factor(SAMPLES$Sentrix_ID)
SAMPLES$CHIP <- factor(SAMPLES$CHIP)
SAMPLES$SUBTYPE <- factor(SAMPLES$SUBTYPE)
SAMPLES$TYPE <- factor(SAMPLES$TYPE)
SAMPLES$SEX <- factor(SAMPLES$SEX)


SAMPLES$TISSUE <- factor(SAMPLES$TISSUE)
SAMPLES$INSTITUTION <- factor(SAMPLES$INSTITUTION)
SAMPLES$STORAGE <- factor(SAMPLES$STORAGE)

################################################################################
stopifnot(all(colnames(EPICM) == SAMPLES$NewCODE)) # all must be TRUE
stopifnot(all(colnames(EPICbeta) == SAMPLES$NewCODE)) # all must be TRUE

# SAMPLES <- SAMPLES[match(colnames(EPICbeta), SAMPLES$NewCODE), ]
# SAMPLES <- EPICM[match(colnames(EPICM), SAMPLES$NewCODE), ]
table(SAMPLES$SUBTYPE)


MARRAY <- as.matrix(EPICM)
colnames(MARRAY) <- colnames(EPICM)
rownames(MARRAY) <- rownames(EPICM)

EPICMraw <- EPICM
EPICbetaraw <- EPICbeta

rm(EPICbeta,EPICM)


################################################################################
################################################################################
# removeBatchEffect
# new way
# https://support.bioconductor.org/p/128357/
# https://support.bioconductor.org/p/83286/
design <- model.matrix( ~0 + SUBTYPE + Sentrix_ID + Sentrix_Position, data=SAMPLES)
design
x <- which(grepl("SUBTYPE",colnames(design)))
treatment.design <- design[,x]
batch.design <- design[,-(x)]
ARRAY_removedBatch_orig <- limma::removeBatchEffect(MARRAY,
design=treatment.design,
covariates=batch.design)

# ARRAY_removedBatch_orig <- limma::removeBatchEffect(MARRAY, batch = SAMPLES$Sentrix_ID)


################################################################################
# remove unwanted TYPES
SAMPLES <- SAMPLES[!TYPE=="MOGHE",]
SAMPLES <- SAMPLES[!TYPE=="HS",]
SAMPLES <- SAMPLES[!TYPE=="complexMCD",]
SAMPLES <- SAMPLES[!SUBTYPE=="FCD1B",]

# SAMPLES <- SAMPLES[!TYPE=="HME",]
# SAMPLES <- SAMPLES[!TYPE=="mMCD",]
# SAMPLES <- SAMPLES[!TYPE=="PMG",]
# SAMPLES <- SAMPLES[!TYPE=="TLE",]
# SAMPLES <- SAMPLES[!TYPE=="TSC",]

SAMPLES$Sentrix_Position <- factor(SAMPLES$Sentrix_Position)
SAMPLES$Sentrix_ID <- factor(SAMPLES$Sentrix_ID)
SAMPLES$CHIP <- factor(SAMPLES$CHIP)
SAMPLES$SUBTYPE <- factor(SAMPLES$SUBTYPE)
SAMPLES$TYPE <- factor(SAMPLES$TYPE)
SAMPLES$SEX <- factor(SAMPLES$SEX)

fwrite(SAMPLES, file = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_met/FCD_EPIC_SAMPLES_final.tsv", sep = "\t")


# remove SAMPLES from ARRAY
ARRAY_removedBatch_orig <- ARRAY_removedBatch_orig[,colnames(ARRAY_removedBatch_orig) %in% SAMPLES$NewCODE]
EPICMraw <- EPICMraw[,colnames(EPICMraw) %in% SAMPLES$NewCODE]
EPICbetaraw <- EPICbetaraw[,colnames(EPICbetaraw) %in% SAMPLES$NewCODE]

################################################################################
################################################################################
# ARRAY without BATCH
EPICM <- as.data.frame(ARRAY_removedBatch_orig)
rownames(EPICM)
colnames(EPICM)

ARRAY_removedBatch <- EPICM
ARRAY_raw <- EPICMraw

save(ARRAY_removedBatch, ARRAY_raw , SAMPLES, file="/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_met/FCD_sesame_February2026_M_removedBatch.Rdata")

############################
#TSV outputs BATCH REMOVED
outputDT <- setDT(as.data.frame(EPICM), keep.rownames = TRUE)
setnames(outputDT,"rn","IlmnID")
fwrite(outputDT, file = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_met/FCD_sesame_Mvalues_February2026_removedBatch.tsv", sep = "\t")
rm(outputDT)
#TSV outputs RAW
outputDT <- setDT(as.data.frame(EPICMraw), keep.rownames = TRUE)
setnames(outputDT,"rn","IlmnID")
fwrite(outputDT, file = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_met/FCD_sesame_Mvalues_February2026_raw.tsv", sep = "\t")
rm(outputDT)
################################################################################
################################################################################
# SAVE BETA raw and normalized
# https://rdrr.io/bioc/lumi/src/R/methylation_preprocessing.R
# ARRAY_raw <- as.data.frame(EPICbeta)
# rownames(ARRAY_raw)
# colnames(ARRAY_raw)


#convert to beta
EPICbeta <- as.data.frame(2^ARRAY_removedBatch_orig/(2^ARRAY_removedBatch_orig+1))
rownames(EPICbeta)
colnames(EPICbeta)

ARRAY_removedBatch <- EPICbeta
ARRAY_raw <- EPICbetaraw

save(ARRAY_removedBatch, ARRAY_raw , SAMPLES, file="/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_met/FCD_sesame_February2026_Beta_removedBatch.Rdata")


############################
#TSV outputs BATCH REMOVED
outputDT <- setDT(as.data.frame(EPICbeta), keep.rownames = TRUE)
setnames(outputDT,"rn","IlmnID")
fwrite(outputDT, file = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_met/FCD_sesame_BETAvalues_February2026_removedBatch.tsv", sep = "\t")
rm(outputDT)
#TSV outputs RAW
outputDT <- setDT(as.data.frame(EPICbetaraw), keep.rownames = TRUE)
setnames(outputDT,"rn","IlmnID")
fwrite(outputDT, file = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_met/FCD_sesame_BETAvalues_February2026_raw.tsv", sep = "\t")
rm(outputDT)