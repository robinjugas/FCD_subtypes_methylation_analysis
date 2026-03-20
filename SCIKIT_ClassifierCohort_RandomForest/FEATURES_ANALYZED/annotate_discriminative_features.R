rm(list = ls())  # vymazu Envrironment
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(GenomicRanges))
suppressPackageStartupMessages(library(annotatr))

load("/home/rj/4TB/PORTABLE_DATA/RDATA/FCD_met/EPICv1ANOT_hg19.Rdata")

################################################################################
#

subtypes <- c("FCD1A","FCD2A","FCD2B","FCD3A","FCD3B","FCD3C","FCD3D")

subtype <- subtypes[1]

for(subtype in subtypes){
  
  FEATURES <- fread(sprintf("discriminative_features_%s.csv",subtype))
  FEATURES <- FEATURES[importance != 0,]
  FEATURES <- FEATURES[direction == "higher_in_class",]
  setnames(FEATURES,"feature","IlmnID")
  
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
  # MERGE CPG and GENE ANNOTATION
  RES[,pos := NULL]
  RES_annotated[,c("seqnames","start","end","mean_in_class", "mean_in_others", "difference", "weighted_score","direction","importance") :=NULL]
  
  
  names(RES)
  names(RES_annotated)
  
  
  
  FINAL <- merge(RES, RES_annotated,by="IlmnID",all=TRUE)
  
  writexl::write_xlsx(FINAL,sprintf("annotated_discriminative_features_%s.xlsx",subtype))
  
}





