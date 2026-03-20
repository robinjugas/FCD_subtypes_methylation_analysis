rm(list = ls())  # vymazu Envrironment
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(GenomicRanges))
suppressPackageStartupMessages(library(annotatr))

################################################################################################################################################################
# 
load("/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_met/EPICv1ANOT_hg19.Rdata")

FEATURES <- fread("selected_features_top8000.csv")
FEATURES <- FEATURES[,1]
setnames(FEATURES,"IlmnID")

FEATURES <- FEATURES[IlmnID %like% "cg"]

################################################################################
## TABLE

RES <- merge(x = FEATURES, y = EPICv1annot, by = "IlmnID", all.x = TRUE)
setnames(RES,c("CHR","MAPINFO"),c("chr","pos"))

RES$chr <- paste0("chr",RES$chr)
RES$chr <- as.factor(RES$chr)
RES$pos <- as.integer(RES$pos)
RES$start <- RES$pos
RES$stop  <- RES$pos #try



dm_regions <- makeGRangesFromDataFrame(RES, keep.extra.columns=TRUE, ignore.strand=TRUE,
                                       seqnames.field=c("seqnames", "seqname",
                                                        "chromosome", "chrom",
                                                        "chr", "chromosome_name",
                                                        "seqid"),
                                       start.field="start",
                                       end.field=c("end", "stop"),
                                       starts.in.df.are.0based=FALSE)

###################################################################################
# Select annotations for intersection with regions
# annots = c('hg19_basicgenes', 'hg19_genes_intergenic', 'hg19_genes_promoters')
# builtin_annotations()
annots = c('hg19_basicgenes') #hg38_genes_promoters  hg38_basicgenes
# Build the annotations (a single GRanges object)
annotations = build_annotations(genome = "hg19", annotations = annots)
# Intersect the regions we read in with the annotations
dm_annotated = annotate_regions(regions = dm_regions, annotations = annotations, ignore.strand = TRUE, quiet = FALSE)
# A GRanges object is returned
RES_annotated <- as.data.table(data.frame(dm_annotated)) #Granges function

names(RES_annotated)

RES_annotated[,c("pos","width","strand","annot.seqnames","annot.start","annot.end","annot.width","annot.strand","annot.id","annot.tx_id","annot.gene_id") :=NULL]

RES_annotated$annot.type <- sub("hg19_*", "", RES_annotated$annot.type)

## sort with preference
# unique(RES_annotated$annot.type)
priority <- c("genes_promoters", "genes_1to5kb", "genes_introns", "genes_exons", "genes_5UTRs")
RES_annotated <- RES_annotated[order(match(annot.type, priority))]
RES_annotated <- unique(RES_annotated, by=c("IlmnID"))



###################################################################################
###################################################################################
# Select annotations for intersection with regions
# annots = c('hg19_basicgenes', 'hg19_genes_intergenic', 'hg19_genes_promoters')
# builtin_annotations()
annots = c('hg19_cpgs') #hg38_genes_promoters  hg38_basicgenes
# Build the annotations (a single GRanges object)
annotations = build_annotations(genome = "hg19", annotations = annots)
# Intersect the regions we read in with the annotations
dm_annotated = annotate_regions(regions = dm_regions, annotations = annotations, ignore.strand = TRUE, quiet = FALSE)
# A GRanges object is returned
RES_annotated_2 <- as.data.table(data.frame(dm_annotated)) #Granges function

names(RES_annotated_2)

RES_annotated_2[,c("pos","width","strand","annot.seqnames","annot.start","annot.end","annot.width","annot.strand","annot.id","annot.tx_id","annot.gene_id") :=NULL]
RES_annotated_2 <- RES_annotated_2[,c("IlmnID","annot.type")]


RES_annotated_2$annot.type <- sub("hg19_*", "", RES_annotated_2$annot.type)

## sort with preference
# unique(RES_annotated_2$annot.type)
priority <- c("cpg_islands", "cpg_shores", "cpg_shelves", "cpg_inter")
RES_annotated_2 <- RES_annotated_2[order(match(annot.type, priority))]
RES_annotated_2 <- unique(RES_annotated_2, by=c("IlmnID"))


###################################################################################
# MERGE CPG and GENE ANNOTATION
RES[,pos := NULL]
RES_annotated[,c("seqnames","start","end","pos" ) :=NULL]
setnames(RES_annotated_2,"annot.type","CpG_annot.type")
setnames(RES_annotated,c("annot.type","annot.symbol"),c("Gene_annot.type","Gene_annot.symbol"))

FINAL <- merge(RES, RES_annotated,by="IlmnID",all=TRUE)
FINAL <- merge(FINAL,RES_annotated_2,by="IlmnID",all=TRUE)




# ################################################################################################################################################################
# # intersection with mixOMICS selected Features
# FEATURESmixOMICS <- fread("/home/rj/ownCloud/PROJECTS/MilanBrazdilMetylace/mixOMICS_2omics_MET+RNA/DIABLO_MET+RNA_design=0.1_FCD+PM/selectedFeatures_final.diablo.model.csv")
# setnames(FEATURESmixOMICS,"IlmnID")
# FEATURESmixOMICS <- FEATURESmixOMICS[IlmnID %like% "cg"]
# X <- intersect(FEATURES$IlmnID,FEATURESmixOMICS$IlmnID)
# 
# 
# FINAL[, inMIXomics := IlmnID %in% X]


writexl::write_xlsx(FINAL,"RF_selectedFeatures_annotated_CpGs.xlsx")

################################################################################################################################################################
# Recode gene body components into a single entity
gene_body_regions <- c("genes_5UTRs", "genes_introns", "genes_exons", "genes_5UTRs")  # adjust to exact strings in your data
FINAL[Gene_annot.type %in% gene_body_regions, Gene_annot.type := "gene body"]


################################################################################################################################################################
FINAL$Gene_annot.type <- sub("genes_*", "", FINAL$Gene_annot.type)
FINAL$CpG_annot.type <- sub("cpg_*", "", FINAL$CpG_annot.type)




# ################################################################################################################################################################
# #barchart 
# library("ggplot2")
# 
# data <- as.data.frame(table(FINAL[,"Gene_annot.type"]))
# colnames(data) <- c("Type","Value")
# 
# 
# # Barplot
# p <- ggplot(data = data, aes(fill=Type, x=1, y=Value))
# p <- p + geom_col(position="fill")
# # p <- p + facet_grid(~samplename)
# p <- p + theme(legend.position='right')
# p <- p + guides(fill=guide_legend(title="Region")) 
# p <- p + scale_y_continuous(labels=c("0.00" = "0","0.25"="25","0.5"="50","0.75"="75","1.00"="100"))
# p <- p + theme(axis.title.y = element_text(size = 6, angle = 90))
# p <- p + theme(axis.title.x = element_text(size = 6, angle = 00))
# p <- p + theme(axis.text.y = element_text(size = 6))
# p <- p + theme(axis.text.x = element_blank())
# p <- p + ylab("Percentage")
# p <- p + xlab("CpGs")
# p <- p + theme(strip.text = element_text(size = 6))
# p <- p + theme(strip.background = element_blank())
# p <- p + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#                panel.background = element_blank(), axis.line = element_blank(),
#                legend.background = element_blank(),legend.key = element_blank(), legend.text = element_text(size=6))
# 
# # print(p)
# 
# ggsave("features_Genes.png", plot = p, height = 40, width = 80, units = "mm",dpi = 300)
# 
# 
# ###################################################################################
# data <- as.data.frame(table(FINAL[,"CpG_annot.type"]))
# colnames(data) <- c("Type","Value")
# 
# 
# # Barplot
# p <- ggplot(data = data, aes(fill=Type, x=1, y=Value))
# p <- p + geom_col(position="fill")
# # p <- p + facet_grid(~samplename)
# p <- p + theme(legend.position='right')
# p <- p + guides(fill=guide_legend(title="Region")) 
# p <- p + scale_y_continuous(labels=c("0.00" = "0","0.25"="25","0.5"="50","0.75"="75","1.00"="100"))
# p <- p + theme(axis.title.y = element_text(size = 6, angle = 90))
# p <- p + theme(axis.title.x = element_text(size = 6, angle = 00))
# p <- p + theme(axis.text.y = element_text(size = 6))
# p <- p + theme(axis.text.x = element_blank())
# p <- p + ylab("Percentage")
# p <- p + xlab("CpGs")
# p <- p + theme(strip.text = element_text(size = 6))
# p <- p + theme(strip.background = element_blank())
# p <- p + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#                panel.background = element_blank(), axis.line = element_blank(),
#                legend.background = element_blank(),legend.key = element_blank(), legend.text = element_text(size=6))
# 
# # print(p)
# 
# ggsave("features_CpGs.png", plot = p, height = 40, width = 80, units = "mm",dpi = 300)
# 



###################################################################################
###################################################################################
library("ggplot2")
library("patchwork")

# --- Helper function to avoid repetition ---
make_bar <- function(df, legend_title, x_label) {
  data <- as.data.frame(table(df))
  colnames(data) <- c("Type", "Value")
  
  ggplot(data = data, aes(fill = Type, x = 1, y = Value)) +
    geom_col(position = "fill") +
    theme(legend.position = 'right') +
    guides(fill = guide_legend(title = legend_title)) +
    scale_y_continuous(labels = c("0.00" = "0", "0.25" = "25", "0.5" = "50", "0.75" = "75", "1.00" = "100")) +
    theme(
      axis.title.y   = element_text(size = 6, angle = 90),
      axis.title.x   = element_text(size = 6, angle = 0),
      axis.text.y    = element_text(size = 6),
      axis.text.x    = element_blank(),
      axis.ticks.x   = element_blank(),
      strip.text     = element_text(size = 6),
      strip.background = element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      axis.line        = element_blank(),
      legend.background = element_blank(),
      legend.key       = element_blank(),
      legend.text      = element_text(size = 6)
    ) +
    ylab("Percentage") +
    xlab(x_label)
}

# --- Build both plots ---
p1 <- make_bar(FINAL[, "Gene_annot.type"], legend_title = "Gene Region", x_label = "")
p2 <- make_bar(FINAL[, "CpG_annot.type"],  legend_title = "CpG Region",  x_label = "")

# --- Combine and save ---
combined <- p1 + p2 + plot_layout(ncol = 2)

ggsave("features_combined.png", plot = combined, height = 40, width = 160, units = "mm", dpi = 300)



