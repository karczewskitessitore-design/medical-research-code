# -*- coding: utf-8 -*-
import io, os, collections, heapq

MAN    = r"E:\医学\01\_pull\kidney_manifest.tsv"
LOCAL  = r"E:\医学\01"
OUT    = r"E:\医学\01\_pull"
NSHARD = 48
BIG    = 256 * 1024 * 1024
NAME_FB_MIN = 1024 * 1024
SKIP_DIRS = ("_pull",)

def rd(p):
    with io.open(p, "r", encoding="utf-8", errors="surrogateescape", newline="") as fh:
        return fh.read()
def wr(p, s):
    with io.open(p, "w", encoding="utf-8", errors="surrogateescape", newline="") as fh:
        fh.write(s)

def tail4(p):
    parts = p.split("/")
    return tuple(parts[-4:]) if len(parts) >= 4 else None

def build_index():
    strong = collections.defaultdict(set)
    exact  = collections.defaultdict(set)
    byname = collections.defaultdict(set)
    n = 0
    for dp, dn, fn in os.walk(LOCAL):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            fp = os.path.join(dp, f)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                continue
            rel = os.path.relpath(fp, LOCAL).replace("\\", "/")
            t = tail4(rel)
            if t is not None:
                strong[sz].add(t)
            exact[sz].add(rel)
            if sz >= NAME_FB_MIN:
                byname[sz].add(f)
            n += 1
    return strong, exact, byname, n

def main():
    files, links = [], []
    for line in rd(MAN).split("\n"):
        line = line.rstrip("\r")
        if not line:
            continue
        f = line.split("\t")
        if len(f) < 3:
            continue
        typ, sz, p = f[0], int(f[1]), f[2]
        if typ == "l":
            links.append((p, f[3] if len(f) > 3 else ""))
        else:
            files.append((sz, p))

    strong, exact, byname, nlocal = build_index()
    print("remote files : %d   (%.3f GB)" % (len(files), sum(s for s, _ in files) / 2 ** 30))
    print("remote links : %d" % len(links))
    print("local  files : %d" % nlocal)

    todo, dup, special = [], [], []
    for sz, p in files:
        if any(ord(c) > 127 for c in p):
            special.append((sz, p)); continue
        t = tail4(p)
        base = p.split("/")[-1]
        if (p in exact.get(sz, ())
                or (t is not None and t in strong.get(sz, ()))
                or (sz >= NAME_FB_MIN and base in byname.get(sz, ()))):
            dup.append((sz, p))
        else:
            todo.append((sz, p))

    small = [r for r in todo if r[0] < BIG]
    big   = [r for r in todo if r[0] >= BIG]

    print("dup   files  : %d   (%.3f GB)" % (len(dup), sum(s for s, _ in dup) / 2 ** 30))
    print("todo  total  : %d   (%.3f GB)" % (len(todo), sum(s for s, _ in todo) / 2 ** 30))
    print("  - small(<256M): %d   (%.3f GB)" % (len(small), sum(s for s, _ in small) / 2 ** 30))
    print("  - big  (>=256M): %d   (%.3f GB)" % (len(big), sum(s for s, _ in big) / 2 ** 30))
    print("special(非ASCII): %d" % len(special))

    mx = max(len(r[1]) for r in small)
    print("max relpath len (small): %d  -> target len approx %d" % (mx, mx + len(r"E:\医学\01\kidney\ ")))

    wr(os.path.join(OUT, "todo_small.tsv"), "".join("%d\t%s\n" % r for r in sorted(small, key=lambda x: -x[0])))
    wr(os.path.join(OUT, "todo_big.tsv"),   "".join("%d\t%s\n" % r for r in sorted(big,   key=lambda x: -x[0])))
    wr(os.path.join(OUT, "dup_list.tsv"),   "".join("%d\t%s\n" % r for r in dup))
    wr(os.path.join(OUT, "special_list.tsv"), "".join("%d\t%s\n" % r for r in special))
    wr(os.path.join(OUT, "symlinks.tsv"),   "".join("%s\t%s\n" % r for r in links))

main()