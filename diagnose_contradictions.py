#!/usr/bin/env python3
"""List the patch pairs where a generated collection disagrees with the
published annotations, with where it happens on the segment.

usage: diagnose_contradictions.py GENERATED.json PATCH_META_DIR PATCH_CACHE UMBILICUS RELATIVE SAME [TOL]
"""
import json
import os
import sys

import numpy as np

# sibling module; python -I drops the script directory from sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
v = __import__("validate_against_patches")


def main():
    sys.stdout.reconfigure(errors="replace")
    gen_path, meta_dir, cache, umb, rel_path, same_path = sys.argv[1:7]
    tol = float(sys.argv[7]) if len(sys.argv) > 7 else 2.5
    gen = v.load_pcl(gen_path)
    allp = np.concatenate([c["p"] for c in gen.values()])
    names, lo, hi = v.patch_bboxes(meta_dir)
    sel = [n for n, l, h in zip(names, lo, hi)
           if np.any(np.all((allp >= l - 10) & (allp <= h + 10), axis=1))]
    ps = v.PatchSet({n: v.fetch_patch(n, "unused", cache) for n in sel})
    center = v.umbilicus_center(umb)
    wf = v.WindingField(ps, center)
    ref = {}
    for label, path in (("relative", rel_path), ("same", same_path)):
        for cid, c in v.load_pcl(path).items():
            hits = ps.nearest(c["p"], tol)
            for key, val in v.pair_claims(v.patch_offsets(c, hits, wf, center, tol)).items():
                ref.setdefault(key, []).append((label, c["name"], round(val, 3)))
    rows = []
    for cid, c in gen.items():
        hits = ps.nearest(c["p"], tol)
        off = v.patch_offsets(c, hits, wf, center, tol)
        # where on the chain each patch is hit
        where = {}
        for n, h in enumerate(hits):
            if h:
                k = min(h, key=lambda kk: h[kk][0])
                if h[k][0] <= tol:
                    where.setdefault(k, []).append(n)
        for (a, b), val in v.pair_claims(off).items():
            if (a, b) not in ref:
                continue
            rv = float(np.median([x[2] for x in ref[(a, b)]]))
            rows.append({
                "collection": c["name"], "n_points": len(c["p"]),
                "confidence": c["meta"].get("confidence"),
                "patch_a": ps.names[a], "patch_b": ps.names[b],
                "chain_idx_a": [min(where[a]), max(where[a])], "chain_idx_b": [min(where[b]), max(where[b])],
                "ours": round(val, 3), "reference": round(rv, 3), "residual": round(val - rv, 3),
                "reference_sources": ref[(a, b)][:4],
                "reference_spread": round(float(np.ptp([x[2] for x in ref[(a, b)]])), 3),
                "verdict": "agree" if abs(val - rv) < 0.5 else "CONTRADICT",
            })
    rows.sort(key=lambda r: (r["verdict"] != "CONTRADICT", r["collection"]))
    print(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
