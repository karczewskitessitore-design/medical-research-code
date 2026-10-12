#!/usr/bin/env Rscript
# ============================================================================
# Exp1 数据准备: GSE183276 (KPMP Kidney scCv3) -> 每细胞类型表达矩阵
#   输入 counts.RDS.gz : 双层 gzip 的 dgCMatrix, 37080 genes x 109741 cells
#        meta.txt.gz   : 引号包裹的 TSV, 第一列为 cell barcode (row.names)
#   输出 data/proc/gse183276/celltype_counts/<CT>.{mtx,genes.txt,cells.txt}
#        data/proc/gse183276/celltype_meta.csv     每细胞: celltype/donor/condition
#        data/proc/gse183276/celltype_summary.csv  每类型: 细胞数/供者数/条件分布
# ============================================================================
suppressMessages({library(Matrix)})

RAW <- "/root/autodl-tmp/vko/data/raw/gse183276"
OUT <- "/root/autodl-tmp/vko/data/proc/gse183276"
CTD <- file.path(OUT, "celltype_counts")
dir.create(CTD, recursive = TRUE, showWarnings = FALSE)

NCELL   <- 1500L    # 每细胞类型最多取多少细胞 (按供者均衡抽样)
MINCELL <- 120L     # 少于该数目则跳过
SEED    <- 1L
set.seed(SEED)

## ---- 细胞类型映射: subclass.l2 -> 工作类型 (同一 label 只归一类) ----------
MAP <- list(
  PCT        = c("PT-S1/2", "PT-S3"),
  PT_injured = c("aPT", "dPT", "cycPT"),
  TAL        = c("C-TAL", "M-TAL", "aTAL1", "aTAL2", "dC-TAL", "dM-TAL"),
  DTL        = c("DTL1", "DTL2", "DTL3"),
  DCT        = c("DCT", "dDCT"),
  CNT        = c("CNT", "dCNT"),
  CD_PC      = c("C-PC", "M-PC", "dM-PC", "tPC-IC"),
  CD_IC      = c("C-IC-A", "dC-IC-A", "M-IC-A", "IC-B", "IMCD"),
  Podocyte   = c("POD", "dPOD"),
  GEnC       = c("EC-GC", "EC-PTC", "EC-AVR", "EC-DVR", "EC-AEA", "cycEC", "dEC"),
  PEC        = c("PEC"),
  Fibroblast = c("FIB", "aFIB", "dFIB", "MYOF", "M-FIB"),
  VSMC_P     = c("VSMC", "VSMC/P", "dVSMC", "REN"),
  Mesangium  = c("MC"),
  T_cell     = c("T"),
  NK_cell    = c("NKC/T", "N", "cycNKC/T"),
  B_Plasma   = c("B", "PL"),
  Myeloid    = c("MAC-M2", "ncMON", "cDC", "cycMNP", "MDC", "MD"),
  Mast       = c("MAST")
)
lab2ct <- character(0)
for (ct in names(MAP)) for (lb in MAP[[ct]]) lab2ct[lb] <- ct

## ---- 1. metadata ----------------------------------------------------------
meta <- read.table(gzfile(file.path(RAW, "meta.txt.gz")), header = TRUE, sep = "\t",
                   row.names = 1, check.names = FALSE, comment.char = "")
cat(sprintf("[meta] %d cells x %d cols\n", nrow(meta), ncol(meta)))
stopifnot(all(c("subclass.l2", "condition.l1", "condition.long", "patient") %in% names(meta)))

## ---- 2. counts (双层 gzip) ------------------------------------------------
x <- readRDS(gzcon(gzfile(file.path(RAW, "counts.RDS.gz"))))
cat(sprintf("[counts] %d genes x %d cells  nnz=%d\n", nrow(x), ncol(x), length(x@x)))

cells <- intersect(colnames(x), rownames(meta))
stopifnot(length(cells) > 10000)
x    <- x[, cells, drop = FALSE]
meta <- meta[cells, , drop = FALSE]
cat(sprintf("[align] %d cells kept\n", length(cells)))

## ---- 3. 基因过滤: 去掉线粒体基因 -------------------------------------------
keep_g <- !grepl("^MT-", rownames(x))
cat(sprintf("[genes] dropped %d MT genes -> %d genes\n", sum(!keep_g), sum(keep_g)))
x <- x[keep_g, , drop = FALSE]

## ---- 4. 标注细胞类型 -------------------------------------------------------
meta$celltype <- unname(lab2ct[as.character(meta[["subclass.l2"]])])
cat(sprintf("[annot] %d/%d cells 获得 celltype 标签\n",
            sum(!is.na(meta$celltype)), nrow(meta)))
print(head(sort(table(meta$celltype, useNA = "ifany"), decreasing = TRUE), 25))

## ---- 5. 供者均衡抽样 -------------------------------------------------------
pick_balanced <- function(idx, donor, n) {
  if (length(idx) <= n) return(idx)
  sp  <- split(idx, donor[idx])
  per <- ceiling(n / length(sp))
  sel <- unlist(lapply(sp, function(v) if (length(v) <= per) v else sample(v, per)),
                use.names = FALSE)
  if (length(sel) > n) sel <- sample(sel, n)
  sel
}

summary_rows <- list(); meta_out <- list()
for (ct in names(MAP)) {
  idx <- which(meta$celltype == ct)
  if (length(idx) < MINCELL) { cat(sprintf("[skip] %-11s n=%d (< %d)\n", ct, length(idx), MINCELL)); next }
  sel <- pick_balanced(idx, meta$patient, NCELL)
  sel <- sel[order(sel)]
  sub <- x[, sel, drop = FALSE]
  sub <- sub[Matrix::rowSums(sub) > 0, , drop = FALSE]
  sel <- sel[rowSums(sub) >= 0]                     # 保持列
  writeMM(sub,  file.path(CTD, paste0(ct, ".mtx")))
  writeLines(rownames(sub), file.path(CTD, paste0(ct, ".genes.txt")))
  writeLines(colnames(sub), file.path(CTD, paste0(ct, ".cells.txt")))
  mm <- data.frame(cell = colnames(sub), celltype = ct,
                   donor = meta$patient[sel],
                   condition = meta$condition.l1[sel],
                   condition_long = meta$condition.long[sel],
                   state = meta$state[sel],
                   stringsAsFactors = FALSE)
  meta_out[[ct]] <- mm
  summary_rows[[ct]] <- data.frame(
    celltype = ct, n_avail = length(idx), n_donors_avail = length(unique(meta$patient[idx])),
    n_used = ncol(sub), n_donors_used = length(unique(mm$donor)),
    genes = nrow(sub),
    n_Ref = sum(mm$condition == "Ref"), n_AKI = sum(mm$condition == "AKI"),
    n_CKD = sum(mm$condition == "CKD"), stringsAsFactors = FALSE)
  cat(sprintf("[ok] %-11s avail=%-6d used=%-5d genes=%-6d donors=%d  Ref/AKI/CKD=%d/%d/%d\n",
              ct, length(idx), ncol(sub), nrow(sub), length(unique(mm$donor)),
              sum(mm$condition == "Ref"), sum(mm$condition == "AKI"), sum(mm$condition == "CKD")))
}
write.csv(do.call(rbind, summary_rows), file.path(OUT, "celltype_summary.csv"), row.names = FALSE)
write.csv(do.call(rbind, meta_out),     file.path(OUT, "celltype_meta.csv"),    row.names = FALSE)
cat("\n[DONE] ->", OUT, "\n")