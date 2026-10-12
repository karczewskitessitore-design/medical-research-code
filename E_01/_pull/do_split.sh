#!/bin/bash
set -u
cd /root/autodl-tmp/kidney || exit 1
mkdir -p _split
: > /root/split_map.tsv
i=0
while IFS=$'\t' read -r sz path; do
  [ -z "$path" ] && continue
  i=$((i+1))
  d=$(printf '_split/%04d' "$i")
  mkdir -p "$d"
  split -b 134217728 -d -a 3 "$path" "$d/p"
  printf '%s\t%s\t%s\n' "$d" "$sz" "$path" >> /root/split_map.tsv
done < /root/bigfiles.tsv
find _split -type f -printf '%s\t%p\n' > /root/parts.tsv
echo "bigfiles=$(wc -l < /root/bigfiles.tsv) parts=$(wc -l < /root/parts.tsv) bytes=$(awk -F"\t" '{s+=$1} END{print s}' /root/parts.tsv)"