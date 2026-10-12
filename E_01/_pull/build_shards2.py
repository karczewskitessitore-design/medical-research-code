import io, os, heapq
OUT = r"E:\医学\01\_pull"
NSHARD = 48
def rd(p):
    with io.open(p, "r", encoding="utf-8", errors="surrogateescape", newline="") as fh: return fh.read()
def wr(p, s):
    with io.open(p, "w", encoding="utf-8", errors="surrogateescape", newline="") as fh: fh.write(s)
rows = []
for name in ("todo_small.tsv", "parts.tsv"):
    for line in rd(os.path.join(OUT, name)).split("\n"):
        line = line.strip("\r")
        if line:
            a, b = line.split("\t", 1); rows.append((int(a), b))
print("tasks=%d  bytes=%.3f GB" % (len(rows), sum(s for s,_ in rows)/2**30))
order = sorted(rows, key=lambda x: -x[0])
bins = [[] for _ in range(NSHARD)]; load = [0]*NSHARD
heap = [(0,i) for i in range(NSHARD)]; heapq.heapify(heap)
for sz, p in order:
    l, i = heapq.heappop(heap); bins[i].append(p); load[i] = l+sz; heapq.heappush(heap,(l+sz,i))
for i in range(NSHARD):
    wr(os.path.join(OUT, "shard_%02d.txt" % i), "".join(p+"\n" for p in bins[i]))
print("shard GB max/min = %.3f / %.3f" % (max(load)/2**30, min(load)/2**30))
print("filecounts:", " ".join(str(len(b)) for b in bins))