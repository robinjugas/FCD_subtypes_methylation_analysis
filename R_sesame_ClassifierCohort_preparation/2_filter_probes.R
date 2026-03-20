rm(list = ls())  # vymazu Envrironment
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# processing
suppressPackageStartupMessages(library(minfi))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(sesame))

################################################################################

load("/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort_February2026_imputed/FCD_sesame_February2026_imputed.Rdata")


################################################################################
#  IMPORT ILLUMINA EPICv1 ANNOT
EPICv1annot <- read.csv("/home/rj/4TB/PORTABLE_DATA/METYLACE_EPIC/ILLUMINA_INFINIUM_COMMERCIAL_FILES/EPICv1/infinium-methylationepic-v-1-0-b5-manifest-file.csv", header = FALSE, sep = ",", quote="\"", dec = ".", skip = 8, fill = TRUE) #slower
header <- read.csv("/home/rj/4TB/PORTABLE_DATA/METYLACE_EPIC/ILLUMINA_INFINIUM_COMMERCIAL_FILES/EPICv1/infinium-methylationepic-v-1-0-b5-manifest-file.csv", header = FALSE, sep = ",", quote="\'", dec = ".", skip = 7, nrow=1, stringsAsFactors = FALSE )
colnames(EPICv1annot) <- header
# EPICv1 <- EPICv1[c("IlmnID","CHR","MAPINFO")]
EPICv1annot <- EPICv1annot[c("IlmnID","CHR_hg38")]
setDT(EPICv1annot)


# #  IMPORT ILLUMINA V2 & get original EPICV1 names
EPICv2annot <- read.csv("/home/rj/4TB/PORTABLE_DATA/METYLACE_EPIC/ILLUMINA_INFINIUM_COMMERCIAL_FILES/EPICv2/EPIC-8v2-0_A1.csv", header = FALSE, sep = ",", quote="\"", dec = ".", skip = 8, fill = TRUE) #slower
header <- read.csv("/home/rj/4TB/PORTABLE_DATA/METYLACE_EPIC/ILLUMINA_INFINIUM_COMMERCIAL_FILES/EPICv2/EPIC-8v2-0_A1.csv", header = FALSE, sep = ",", quote="\'", dec = ".", skip = 7, nrow=1, stringsAsFactors = FALSE )
colnames(EPICv2annot) <- header
EPICv2annot <- EPICv2annot[c("CHR","EPICv1_Loci")]
setDT(EPICv2annot)
setnames(EPICv2annot,"EPICv1_Loci","IlmnID") # cause I use betasCollapseToPfx

################################################################################
################################################################################
XYprobes1 <- EPICv1annot[((CHR_hg38=="chrY") | (CHR_hg38=="chrX")),"IlmnID"]
XYprobes2 <- EPICv2annot[((CHR=="chrY") | (CHR=="chrX")),"IlmnID"]
XYprobes <- unique(c(XYprobes1$IlmnID,XYprobes2$IlmnID))


EPICbeta <- EPICbeta[!(rownames(EPICbeta) %in% XYprobes),]
EPICM <- EPICM[!(rownames(EPICM) %in% XYprobes),]

################################################################################
# https://github.com/sirselim/illumina450k_filtering
## generate 'bad' probes filter
# cross-reactive/non-specific
cross.react <- read.csv('/home/rj/4TB/PORTABLE_DATA/METYLACE_EPIC/IlluminaEPIC_probes2filter/48639-non-specific-probes-Illumina450k.csv', head = T, as.is = T)
cross.react.probes <- as.character(cross.react$TargetID)
# BOWTIE2 multi-mapped
multi.map <- read.csv('/home/rj/4TB/PORTABLE_DATA/METYLACE_EPIC/IlluminaEPIC_probes2filter/HumanMethylation450_15017482_v.1.1_hg19_bowtie_multimap.txt', head = F, as.is = T)
multi.map.probes <- as.character(multi.map$V1)


# probes from Pidsley 2016 (EPIC)
epic.cross1 <- read.csv('/home/rj/4TB/PORTABLE_DATA/METYLACE_EPIC/IlluminaEPIC_probes2filter/EPIC/13059_2016_1066_MOESM1_ESM.csv', head = T)
# epic.cross2 <- read.csv('/home/rj/4TB/PORTABLE_DATA/METYLACE_EPIC/IlluminaEPIC_probes2filter/EPIC/13059_2016_1066_MOESM2_ESM.csv', head = T)
# epic.cross3 <- read.csv('/home/rj/4TB/PORTABLE_DATA/METYLACE_EPIC/IlluminaEPIC_probes2filter/EPIC/13059_2016_1066_MOESM3_ESM.csv', head = T)
epic.variants1 <- read.csv('/home/rj/4TB/PORTABLE_DATA/METYLACE_EPIC/IlluminaEPIC_probes2filter/EPIC/13059_2016_1066_MOESM4_ESM.csv', head = T)
epic.variants2 <- read.csv('/home/rj/4TB/PORTABLE_DATA/METYLACE_EPIC/IlluminaEPIC_probes2filter/EPIC/13059_2016_1066_MOESM5_ESM.csv', head = T)
epic.variants3 <- read.csv('/home/rj/4TB/PORTABLE_DATA/METYLACE_EPIC/IlluminaEPIC_probes2filter/EPIC/13059_2016_1066_MOESM6_ESM.csv', head = T)


OMIT.probes <- unique(c(cross.react.probes,multi.map.probes,
                        as.character(epic.cross1$X), as.character(epic.variants1$PROBE), 
                        as.character(epic.variants2$PROBE), as.character(epic.variants3$PROBE)))


EPICbeta <- EPICbeta[!(rownames(EPICbeta) %in% OMIT.probes),]
EPICM <- EPICM[!(rownames(EPICM) %in% OMIT.probes),]


################################################################################
# remove non CpG probes aka chr probes https://knowledge.illumina.com/microarray/general/microarray-general-troubleshooting-list/000005501
# ch.8.128185533F
# EPICbeta <- EPICbeta[!grep("^ch", rownames(EPICbeta)),]
# grep("^ch",rownames(EPICbeta))

#remove probes without cg
EPICbeta <- EPICbeta[grepl("^cg", rownames(EPICbeta)), ]
EPICM <- EPICM[grepl("^cg", rownames(EPICM)), ]

grepl("^cg",rownames(EPICbeta))
grepl("^cg",rownames(EPICM))


################################################################################
#https://stackoverflow.com/questions/75270932/remove-all-rows-from-data-table-if-there-is-any-infinite-value
# remove NAs and Inf for DT
# ARRAY <- ARRAY[ARRAY[, Reduce(`&`, lapply(.SD, is.finite)), .SDcols = is.numeric]]

EPICbeta <- na.omit(EPICbeta)
EPICM <- na.omit(EPICM)

################################################################################
# EXCEL for control
# wb <- openxlsx::createWorkbook()
# openxlsx::addWorksheet(wb, sheetName = "samples")
# openxlsx::writeData(wb, sheet = "samples", as.data.frame(SAMPLES))
# openxlsx::saveWorkbook(wb, "SAMPLES_used.xlsx", overwrite = TRUE, returnValue = FALSE)

##
save(EPICbeta, EPICM, SAMPLES, file="/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort_February2026_imputed/FCD_sesame_February2026_imputed_filteredprobes.Rdata", compress=TRUE)

################################################################################
# TSV
EPICbetaDT <- setDT(EPICbeta, keep.rownames = TRUE)
setnames(EPICbeta,"rn","IlmnID")

EPICMDT <- setDT(EPICM, keep.rownames = TRUE)
setnames(EPICMDT,"rn","IlmnID")


# fwrite(SAMPLES, file = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort/FCD_EPIC_SAMPLES.tsv", sep = "\t")
# fwrite(EPICbetaDT, file = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort/FCD_sesame_BETAvalues_October2025_filteredprobes.tsv", sep = "\t")
# fwrite(EPICMDT, file = "/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_MLcohort/FCD_sesame_Mvalues_October2025_filteredprobes.tsv", sep = "\t")





