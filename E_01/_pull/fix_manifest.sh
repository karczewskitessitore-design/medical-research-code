#!/bin/bash
set -euo pipefail
: > /tmp/kidney_chunks/manifest.tsv
for f in /tmp/kidney_chunks/c*.txt; do
  id=$(basename "$f" .txt)
  b=$(awk -F"\t" -v F="$f" 'NR==FNR{s[$2]=$1+0;next}{t+=s[$1]}END{printf "%d",t+0}' /root/all_sorted.tsv "$f")
  printf "%s\t%s\n" "$id" "$b" >> /tmp/kidney_chunks/manifest.tsv
done
echo "chunks=$(ls /tmp/kidney_chunks/c*.txt | wc -l)"
awk -F"\t" '{s+=$2; if($2<mx||mx==0)mn=$2; if($2>mx)mx=$2} END{printf "total=%.2f GB  min=%.1f MB  max=%.1f MB\n", s/1073741824, mn/1048576, mx/1048576}' /tmp/kidney_chunks/manifest.tsv
head -3 /tmp/kidney_chunks/manifest.tsv
echo "--- 每块文件数抽样 ---"; wc -l /tmp/kidney_chunks/c0000.txt /tmp/kidney_chunks/c0400.txt