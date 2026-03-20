rm(list = ls())  # vymazu Envrironment
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# processing
suppressPackageStartupMessages(library(minfi))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(limma))
suppressPackageStartupMessages(library(sva))


################################################################################
load("/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort_February2026_imputed/FCD_sesame_February2026_imputed_filteredprobes.Rdata")

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

################################################################################
# remove unwanted TYPES
SAMPLES <- SAMPLES[!TYPE=="MOGHE",]
SAMPLES <- SAMPLES[!TYPE=="HS",]
SAMPLES <- SAMPLES[!TYPE=="complexMCD",]
SAMPLES <- SAMPLES[!SUBTYPE=="FCD1B",]

SAMPLES <- SAMPLES[!TYPE=="HME",]
SAMPLES <- SAMPLES[!TYPE=="mMCD",]
SAMPLES <- SAMPLES[!TYPE=="PMG",]
SAMPLES <- SAMPLES[!TYPE=="TLE",]
SAMPLES <- SAMPLES[!TYPE=="TSC",]


# remove SAMPLES from ARRAY
ARRAY_removedBatch_orig <- ARRAY_removedBatch_orig[,colnames(ARRAY_removedBatch_orig) %in% SAMPLES$NewCODE]
EPICMraw <- EPICMraw[,colnames(EPICMraw) %in% SAMPLES$NewCODE]
EPICbetaraw <- EPICbetaraw[,colnames(EPICbetaraw) %in% SAMPLES$NewCODE]

################################################################################
# SAMPLES[,c("Sample_Plate" ,"Sample_Group","Pool_ID","Project","Sample_Well") :=NULL]
outputDT <- SAMPLES
setDT(outputDT)
setnames(outputDT,"NewCODE" ,"Sample")
setcolorder(outputDT,"Sample",before=1)
fwrite(outputDT, file = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort_February2026_imputed/FCD_EPIC_SAMPLES_February2026_forPython.tsv", sep = "\t")
rm(outputDT)

################################################################################
################################################################################
# ARRAY without BATCH
EPICM <- as.data.frame(ARRAY_removedBatch_orig)
rownames(EPICM)
colnames(EPICM)

## for Rdata
# ARRAY_removedBatch <- EPICM
# ARRAY_raw <- EPICMraw
# save(ARRAY_removedBatch, ARRAY_raw , SAMPLES, file="/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort/FCD_sesame_October2025_M_removedBatch.Rdata")

############################
#TSV TRANSPOSE outputs BATCH REMOVED

outputDT <- transpose(EPICM)
setDT(outputDT)
setnames(outputDT, rownames(EPICM))
outputDT[,"Sample" := colnames(EPICM)]
setcolorder(outputDT,"Sample",before=1)
fwrite(outputDT, file = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort_February2026_imputed/FCD_sesame_Mvalues_February2026_removedBatch_forPython.tsv", sep = "\t")
rm(outputDT)

#TSV outputs RAW
outputDT <- transpose(EPICMraw)
setDT(outputDT)
setnames(outputDT, rownames(EPICMraw))
outputDT[,"Sample" := colnames(EPICMraw)]
setcolorder(outputDT,"Sample",before=1)
fwrite(outputDT, file = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort_February2026_imputed/FCD_sesame_Mvalues_February2026_raw_forPython.tsv", sep = "\t")
rm(outputDT)
################################################################################
################################################################################
# SAVE BETA raw and normalized


#convert to beta
EPICbeta <- as.data.frame(2^ARRAY_removedBatch_orig/(2^ARRAY_removedBatch_orig+1))
rownames(EPICbeta)
colnames(EPICbeta)

## for Rdata
# ARRAY_removedBatch <- EPICbeta
# ARRAY_raw <- EPICbetaraw
# save(ARRAY_removedBatch, ARRAY_raw , SAMPLES, file="/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort/FCD_sesame_December2025_Beta_removedBatch_SentrixID_forPython.Rdata")


############################
#TSV outputs BATCH REMOVED
outputDT <- transpose(EPICbeta)
setDT(outputDT)
setnames(outputDT, rownames(EPICbeta))
outputDT[,"Sample" := colnames(EPICbeta)]
setcolorder(outputDT,"Sample",before=1)
fwrite(outputDT, file = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort_February2026_imputed/FCD_sesame_BETAvalues_February2026_removedBatch_forPython.tsv", sep = "\t")
rm(outputDT)
#TSV outputs RAW
outputDT <- transpose(EPICbetaraw)
setDT(outputDT)
setnames(outputDT, rownames(EPICbetaraw))
outputDT[,"Sample" := colnames(EPICbetaraw)]
setcolorder(outputDT,"Sample",before=1)
fwrite(outputDT, file = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort_February2026_imputed/FCD_sesame_BETAvalues_February2026_raw_forPython.tsv", sep = "\t")
rm(outputDT)