#!/usr/bin/env Rscript
# ============================================================================
# Exp1 单细胞类型虚拟敲除 (scTenifoldKnk v1.1, transcriptomeWide)
#   与 Exp0 的 run_vko.R 相同流程, 但打分同时输出:
#     self_dist   对角元 (该基因被敲除后自身扰动距离)
#     mean_offdiag 去对角后的平均扰动距离  <-- 建议主指标
#     sum_offdiag  去对角后的总扰动距离
#   并保存本次实际使用的细胞及其供者/条件 (给匹配零假设与供者 bootstrap 用)
# ============================================================================
suppressMessages({library(Matrix); library(scTenifoldKnk); library(RhpcBLASctl)})
blas_set_num_threads(1); omp_set_num_threads(1)

a <- commandArgs(trailingOnly = TRUE)
g <- function(n, d) { i <- which(a == paste0("--", n)); if (length(i)) a[i + 1] else d }

ct     <- g("ct", "PCT")
base   <- g("base", "/root/autodl-tmp/vko/data/proc/gse183276/celltype_counts")
out    <- g("out",  "/root/autodl-tmp/vko/results/exp1")
ngene  <- as.integer(g("ngene",  "2500"))
ncell  <- as.integer(g("ncell",  "1500"))
nNet   <- as.integer(g("nNet",   "10"))
nCells <- as.integer(g("nCells", "500"))
nCores <- as.integer(g("nCores", "1"))
seed   <- as.integer(g("seed",   "1"))

dir.create(out, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

X <- readMM(file.path(base, paste0(ct, ".mtx")))
rownames(X) <- readLines(file.path(base, paste0(ct, ".genes.txt")))
colnames(X) <- readLines(file.path(base, paste0(ct, ".cells.txt")))
cat(sprintf("[in] %s: %d genes x %d cells\n", ct, nrow(X), ncol(X)))

if (ncol(X) > ncell) X <- X[, sample(ncol(X), ncell)]
lib <- colSums(X); X <- X[, lib > 0, drop = FALSE]; lib <- lib[lib > 0]
Xn <- log1p(sweep(as.matrix(X), 2, lib, "/") * 1e4)
v  <- apply(Xn, 1, var)
X  <- X[order(v, decreasing = TRUE)[seq_len(min(ngene, nrow(X)))], , drop = FALSE]
rm(Xn); gc()
cat(sprintf("[hvg] %d genes x %d cells\n", nrow(X), ncol(X)))

write.csv(data.frame(cell = colnames(X)), file.path(out, paste0(ct, "_used_cells.csv")), row.names = FALSE)

t0  <- Sys.time()
res <- scTenifoldKnk(countMatrix = X, transcriptomeWide = TRUE,
                     qc = FALSE, nc_nNet = nNet, nc_nCells = nCells, nCores = nCores)
el  <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

pd <- as.matrix(res$perturbationDistances)
saveRDS(res, file.path(out, paste0(ct, "_scTenifoldKnk.rds")))
write.csv(pd, file.path(out, paste0(ct, "_perturbationDistances.csv")))

d  <- diag(pd)
m2 <- pd; diag(m2) <- NA
sm <- data.frame(gene         = rownames(pd),
                 self_dist    = d,
                 mean_offdiag = rowMeans(m2, na.rm = TRUE),
                 sum_offdiag  = rowSums(m2,  na.rm = TRUE),
                 mean_all     = rowMeans(pd, na.rm = TRUE),
                 n_na         = rowSums(is.na(pd)),
                 stringsAsFactors = FALSE)
sm <- sm[order(sm$mean_offdiag, decreasing = TRUE), ]
sm$rank_offdiag <- seq_len(nrow(sm))
write.csv(sm, file.path(out, paste0(ct, "_gene_summary.csv")), row.names = FALSE)

cat(sprintf("[TIME] %s  G=%d  cells=%d  nNet=%d  cores=%d  ->  %.2f min\n",
            ct, nrow(X), ncol(X), nNet, nCores, el))
cat("[TOP 15 by mean_offdiag]\n"); print(head(sm[, c("gene","self_dist","mean_offdiag")], 15))
cat("[DONE]", ct, "\n")