#!/bin/bash
set -euo pipefail
cd /root
cat /root/todo_small.tsv /root/parts.tsv > /root/all_files.tsv
LC_ALL=C sort -t$'\t' -k2,2 /root/all_files.tsv > /root/all_sorted.tsv
wc -l /root/all_sorted.tsv
rm -f /tmp/kidney_chunks/c*.txt /tmp/kidney_chunks/manifest.tsv
mkdir -p /tmp/kidney_chunks
awk -F'\t' -v lim=157286400 -v out="/tmp/kidney_chunks" '
NR==FNR { sz[$2]=$1+0; next }
{
  s = sz[$2]
  if (cur > 0 && cur + s > lim) { close(f); idx++; cur = 0 }
  if (cur == 0) { f = sprintf("%s/c%04d.txt", out, idx); bytes=0 }
  print $2 > f
  cur += s; bytes += s
}
END {
  close(f)
  for (i = 0; i <= idx; i++) printf "%04d\t%d\n", i, 0 > "/dev/null"
  printf "%d\n", idx+1
}' /root/all_sorted.tsv /root/all_sorted.tsv
# 重算每块字节数
: > /tmp/kidney_chunks/manifest.tsv
for f in /tmp/kidney_chunks/c*.txt; do
  id=$(basename "$f" .txt)
  b=$(awk -F"\t" -v f="$f" 'NR==FNR{s[$2]=$1;next}{t+=s[$2]}END{printf "%d",t}' /root/all_sorted.tsv "$f")
  printf "%s\t%s\n" "$id" "$b" >> /tmp/kidney_chunks/manifest.tsv
done
echo "chunks=$(ls /tmp/kidney_chunks/c*.txt | wc -l)"
echo "total_bytes=$(awk -F"\t" "{s+=\$2} END{printf \"%d\",s}" /tmp/kidney_chunks/manifest.tsv)"
head -3 /tmp/kidney_chunks/manifest.tsv
du -sh /tmp/kidney_chunks