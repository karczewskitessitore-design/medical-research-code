suppressMessages(library(Matrix))
setwd("/root/autodl-tmp/vko/results/exp0")
cts <- c("PCT","PT_injured","TAL","DCT","CD_PC","Podocyte","GEnC","Mesangium","Fibroblast")
top <- list(); info <- list()
for (ct in cts) {
  m <- as.matrix(read.csv(paste0(ct,"_perturbationDistances.csv"), row.names=1, check.names=FALSE))
  d <- diag(m)
  m2 <- m; diag(m2) <- NA
  r_ex <- rowMeans(m2, na.rm=TRUE)
  r_in <- rowMeans(m,  na.rm=TRUE)
  o <- order(r_ex, decreasing=TRUE)
  o2 <- order(r_in, decreasing=TRUE)
  info[[ct]] <- data.frame(ct=ct, G=nrow(m), diag_mean=mean(d), offdiag_mean=mean(m2,na.rm=TRUE),
                           diag_share=mean(d)/(mean(d)+mean(m2,na.rm=TRUE)),
                           spearman=cor(r_ex, r_in, method="spearman"), stringsAsFactors=FALSE)
  top[[ct]] <- rownames(m2)[o[1:50]]
  cat("\n=====", ct, "=====\n")
  cat("top20 (excl. diagonal):", paste(rownames(m2)[o[1:20]], collapse=", "), "\n")
}
cat("\n=== diag stats ===\n"); print(do.call(rbind, info), digits=4)
M <- sapply(top, function(x) x[1:50])
cat("\n=== pairwise top50 overlap (count) ===\n")
ov <- outer(1:9,1:9,Vectorize(function(i,j) length(intersect(M[,i],M[,j]))))
dimnames(ov) <- list(cts,cts); print(ov)
cat("\n=== how many of the 9 lists contain gene (>=5) ===\n")
tb <- table(unlist(top)); print(sort(tb[tb>=5], decreasing=TRUE)[1:25])