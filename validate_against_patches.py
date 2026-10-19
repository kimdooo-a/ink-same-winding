#!/usr/bin/env python3
"""Check generated same-winding collections against verified surface patches.

Metrics (all distances in the json's voxel units):

  attach rate      fraction of generated points within T voxels of any
                   verified patch surface (exact point-to-triangle distance on
                   the patch's quad grid, triangulated)
  multi-patch      collections whose points land on >= 2 distinct patches;
                   these are the ones that add cross-patch (delta-0) edges
  contradictions   each collection implies a real offset between the continuous
                   winding fields (W = unwrapped theta / 2pi) of the patches it lands
                   on; the same offset is derived from the published annotations
                   (relative_windings.json, same_windings.json); a difference of
                   >= 0.5 turn is a contradiction. Also: a chain that lands on one
                   patch at several places must give one offset (self-consistency).
  angle check      unwrapped angle around the umbilicus along each collection:
                   largest step between consecutive points (spiral-fitting
                   assumes |dtheta| < pi) and total span in turns

usage:
  validate_against_patches.py GENERATED.json --patch-meta-dir DIR --patch-root URL_OR_DIR
      --patch-cache DIR --umbilicus umbilicus.json
      [--relative relative_windings.json --same same_windings.json] [--out report.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None


def load_pcl(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for cid, c in d["collections"].items():
        items = sorted(c["points"].items(), key=lambda kv: int(kv[0]))
        if not items:
            continue
        p = np.array([v["p"] for _, v in items], float)  # [x, y, z]
        w = np.array([np.nan if v.get("wind_a") is None else float(v["wind_a"]) for _, v in items])
        out[int(cid)] = {"name": c["name"], "p": p, "wind": w, "meta": c.get("metadata", {})}
    return out


def patch_bboxes(meta_dir):
    names, lo, hi = [], [], []
    for f in glob.glob(os.path.join(glob.escape(meta_dir), "*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                m = json.load(fh)
            b = np.asarray(m["bbox"], float)
        except Exception:
            continue
        names.append(os.path.basename(f)[:-5])
        lo.append(b[0])
        hi.append(b[1])
    return names, np.asarray(lo), np.asarray(hi)


def fetch_patch(name, root, cache):
    d = os.path.join(cache, name)
    os.makedirs(d, exist_ok=True)
    for c in "xyz":
        dst = os.path.join(d, f"{c}.tif")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        src = f"{root.rstrip('/')}/{name}/{c}.tif"
        if os.path.isdir(root):
            src = os.path.join(root, name, f"{c}.tif")
            with open(src, "rb") as fi, open(dst + f".part{os.getpid()}", "wb") as fo:
                fo.write(fi.read())
        else:
            urllib.request.urlretrieve(src, dst + f".part{os.getpid()}")
        os.replace(dst + f".part{os.getpid()}", dst)
    xyz = np.stack([tifffile.imread(os.path.join(d, f"{c}.tif")).astype(np.float32) for c in "xyz"], -1)
    return xyz


class PatchSet:
    """Vertices of many tifxyz patches + exact distance to their triangles."""

    def __init__(self, patches):
        self.names = list(patches)
        self.grids = [patches[n] for n in self.names]
        verts, owner, ij = [], [], []
        for k, g in enumerate(self.grids):
            v = np.all(g > 0, -1)
            ii, jj = np.nonzero(v)
            verts.append(g[ii, jj])
            owner.append(np.full(len(ii), k))
            ij.append(np.c_[ii, jj])
        self.verts = np.concatenate(verts)
        self.owner = np.concatenate(owner)
        self.ij = np.concatenate(ij)
        self.tree = cKDTree(self.verts)

    def nearest(self, p, radius):
        """For each point: dict patch_index -> (exact distance, i, j) within radius;
        (i, j) is the patch grid cell of the nearest vertex."""
        res = []
        cand = self.tree.query_ball_point(p, r=radius + 30.0)
        for q, idx in zip(p, cand):
            best = {}
            if idx:
                idx = np.asarray(idx)
                # nearest vertex per patch, then exact distance on its quads
                d = np.linalg.norm(self.verts[idx] - q, axis=1)
                for k in np.unique(self.owner[idx]):
                    sel = idx[self.owner[idx] == k]
                    v = sel[np.argmin(d[self.owner[idx] == k])]
                    dist = self._quad_distance(q, k, *self.ij[v])
                    if dist <= radius:
                        best[int(k)] = (dist, int(self.ij[v][0]), int(self.ij[v][1]))
            res.append(best)
        return res

    def _quad_distance(self, q, k, i, j):
        g = self.grids[k]
        h, w = g.shape[:2]
        dmin = np.inf
        for di in (-1, 0):
            for dj in (-1, 0):
                i0, j0 = i + di, j + dj
                if i0 < 0 or j0 < 0 or i0 + 1 >= h or j0 + 1 >= w:
                    continue
                a, b, c, e = g[i0, j0], g[i0, j0 + 1], g[i0 + 1, j0], g[i0 + 1, j0 + 1]
                if min(a[2], b[2], c[2], e[2]) <= 0:
                    continue
                dmin = min(dmin, tri_dist(q, a, b, c), tri_dist(q, b, e, c))
        if not np.isfinite(dmin):
            dmin = float(np.linalg.norm(g[i, j] - q))
        return float(dmin)


def tri_dist(p, a, b, c):
    """Exact point-triangle distance (Ericson, Real-Time Collision Detection)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return np.linalg.norm(ap)
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return np.linalg.norm(bp)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return np.linalg.norm(p - (a + v * ab))
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return np.linalg.norm(cp)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return np.linalg.norm(p - (a + w * ac))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return np.linalg.norm(p - (b + w * (c - b)))
    denom = 1.0 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    return np.linalg.norm(p - (a + ab * v + ac * w))


def umbilicus_center(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    cp = sorted(d["control_points"], key=lambda q: q["z"])
    z = np.array([q["z"] for q in cp], float)
    x = np.array([q["x"] for q in cp], float)
    y = np.array([q["y"] for q in cp], float)
    return lambda zz: (np.interp(zz, z, x), np.interp(zz, z, y))


def angle_stats(p, center):
    cx, cy = center(p[:, 2])
    th = np.unwrap(np.arctan2(p[:, 1] - cy, p[:, 0] - cx))
    steps = np.abs(np.diff(np.arctan2(p[:, 1] - cy, p[:, 0] - cx)))
    steps = np.minimum(steps, 2 * np.pi - steps)
    r = np.hypot(p[:, 0] - cx, p[:, 1] - cy)
    return float(steps.max() if len(steps) else 0.0), float((th.max() - th.min()) / (2 * np.pi)), float(np.median(r))


def patch_theta(grid, center):
    """Angle around the umbilicus, unwrapped over the patch grid (BFS over
    4-neighbours), so it is continuous along the sheet the patch covers."""
    from collections import deque
    valid = np.all(grid > 0, -1)
    cx, cy = center(grid[..., 2])
    raw = np.arctan2(grid[..., 1] - cy, grid[..., 0] - cx)
    out = np.full(raw.shape, np.nan)
    h, w = raw.shape
    for seed in zip(*np.nonzero(valid)):
        if not np.isnan(out[seed]):
            continue
        out[seed] = raw[seed]
        dq = deque([seed])
        while dq:
            i, j = dq.popleft()
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < h and 0 <= b < w and valid[a, b] and np.isnan(out[a, b]):
                    d = raw[a, b] - raw[i, j]
                    d = (d + np.pi) % (2 * np.pi) - np.pi
                    out[a, b] = out[i, j] + d
                    dq.append((a, b))
    return out


class WindingField:
    """Continuous winding coordinate W = theta_unwrapped / (2 pi) (+ const).

    Measured on this scroll: radius grows with increasing atan2(y, x)
    (outward = +theta), and wind_a grows outward, so W = theta / 2pi and a
    relative annotation's integer steps add to W directly.
    """

    def __init__(self, ps, center):
        self.ps, self.center, self.cache = ps, center, {}

    def patch_w(self, k, i, j):
        if k not in self.cache:
            self.cache[k] = patch_theta(self.ps.grids[k], self.center) / (2 * np.pi)
        return self.cache[k][i, j]


def chain_w(p, center, wind=None):
    """W along one collection, up to a constant. Same-winding chains are
    unwrapped along point order; radial relative annotations add wind_a."""
    cx, cy = center(p[:, 2])
    th = np.arctan2(p[:, 1] - cy, p[:, 0] - cx)
    if wind is not None and np.isfinite(wind).all() and np.ptp(wind) > 0:
        # relative annotation: points are radial (|dtheta| ~ 0.01 rad); integer
        # winding steps plus the small wrapped angle offset to the first point
        d = (th - th[0] + np.pi) % (2 * np.pi) - np.pi
        return (wind - wind[0]) + d / (2 * np.pi)
    return np.unwrap(th) / (2 * np.pi)


def patch_offsets(c, hits, wf, center, tol):
    """patch index -> list of (W_chain - W_patch) at the points that attach."""
    wc = chain_w(c["p"], center, c["wind"])
    off = {}
    for n, h in enumerate(hits):
        if not h:
            continue
        k = min(h, key=lambda kk: h[kk][0])
        dist, i, j = h[k]
        if dist > tol:
            continue
        wp = wf.patch_w(k, i, j)
        if np.isfinite(wp):
            off.setdefault(k, []).append(wc[n] - wp)
    return off


def pair_claims(off):
    """(a, b) -> claimed (k_a - k_b): the real offset between two patches'
    winding fields that this collection implies."""
    med = {k: float(np.median(v)) for k, v in off.items()}
    ks = sorted(med)
    return {(a, b): med[a] - med[b] for ai, a in enumerate(ks) for b in ks[ai + 1:]}


def _dmin(h):
    return min(v[0] for v in h.values()) if h else np.inf


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("generated")
    ap.add_argument("--patch-meta-dir", required=True)
    ap.add_argument("--patch-root", required=True)
    ap.add_argument("--patch-cache", required=True)
    ap.add_argument("--umbilicus", required=True)
    ap.add_argument("--relative")
    ap.add_argument("--same")
    ap.add_argument("--tol", type=float, nargs="+", default=[2.5, 5.0, 10.0])
    ap.add_argument("--offset", type=float, nargs=3, default=[0, 0, 0],
                    help="xyz offset added to generated points (frame test)")
    ap.add_argument("--min-patch-points", type=int, default=1,
                    help="a generated collection must touch a patch with at least this many "
                         "points before its offset to that patch is used")
    ap.add_argument("--radial-shift", type=float, default=0.0,
                    help="negative control: move every generated point this many voxels "
                         "radially outward from the umbilicus (about one winding = 19)")
    ap.add_argument("--out")
    args = ap.parse_args()

    gen = load_pcl(args.generated)
    center0 = umbilicus_center(args.umbilicus)
    for c in gen.values():
        c["p"] = c["p"] + np.asarray(args.offset)
        if args.radial_shift:
            cx, cy = center0(c["p"][:, 2])
            d = np.c_[c["p"][:, 0] - cx, c["p"][:, 1] - cy]
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            c["p"][:, :2] += args.radial_shift * d
    allp = np.concatenate([c["p"] for c in gen.values()])
    tmax = max(args.tol)
    names, lo, hi = patch_bboxes(args.patch_meta_dir)
    print(f"patch metas: {len(names)}", flush=True)
    sel = [n for n, l, h in zip(names, lo, hi)
           if np.any(np.all((allp >= l - tmax) & (allp <= h + tmax), axis=1))]
    print(f"candidate patches overlapping generated points: {len(sel)}", flush=True)
    with ThreadPoolExecutor(16) as ex:
        grids = dict(zip(sel, ex.map(lambda n: fetch_patch(n, args.patch_root, args.patch_cache), sel)))
    ps = PatchSet(grids)
    center = umbilicus_center(args.umbilicus)
    wf = WindingField(ps, center)

    # reference relations from the published annotations (restricted to the
    # patch set near the generated points)
    ref = {}
    for label, path in (("relative", args.relative), ("same", args.same)):
        if not path:
            continue
        for c in load_pcl(path).values():
            hits = ps.nearest(c["p"], args.tol[0])
            for key, val in pair_claims(patch_offsets(c, hits, wf, center, args.tol[0])).items():
                ref.setdefault(key, []).append((label, val))
    print(f"reference patch pairs: {len(ref)}", flush=True)

    hits = {cid: ps.nearest(c["p"], tmax) for cid, c in gen.items()}
    dists = np.array([_dmin(h) for hs in hits.values() for h in hs])
    fin = np.isfinite(dists)
    report = {"generated": os.path.basename(args.generated), "points": len(allp),
              "collections": len(gen), "candidate_patches": len(sel), "offset": args.offset, "radial_shift": args.radial_shift, "min_patch_points": args.min_patch_points,
              "reference_patch_pairs": len(ref),
              "distance_percentiles_attached_p10_p50_p90": (
                  np.percentile(dists[fin], [10, 50, 90]).round(3).tolist() if fin.any() else None),
              "per_tol": {}, "per_collection": []}
    for t in args.tol:
        kinds = {"line": [0, 0], "gap": [0, 0]}
        multi = 0
        self_spread = []
        cmp_ = {"agree": 0, "contradict": 0, "pairs_without_reference": 0}
        residuals = []
        for cid, c in gen.items():
            kind = "gap" if c["meta"].get("evidence") == "intercolumnar_gap" else "line"
            ds = np.array([_dmin(h) for h in hits[cid]])
            kinds[kind][0] += int((ds <= t).sum())
            kinds[kind][1] += len(ds)
            off = patch_offsets(c, hits[cid], wf, center, t)
            off = {k: vv for k, vv in off.items() if len(vv) >= args.min_patch_points}
            # self-consistency: a same-winding chain landing on one patch at
            # several places must give the same offset each time
            for v in off.values():
                if len(v) > 1:
                    self_spread.append(float(np.ptp(v)))
            if len(off) >= 2:
                multi += 1
            for key, val in pair_claims(off).items():
                if key not in ref:
                    cmp_["pairs_without_reference"] += 1
                    continue
                rv = float(np.median([x[1] for x in ref[key]]))
                residuals.append(val - rv)
                cmp_["agree" if abs(val - rv) < 0.5 else "contradict"] += 1
        ss = np.asarray(self_spread)
        report["per_tol"][str(t)] = {
            "attach_rate": float((dists <= t).mean()),
            "attach_rate_lines": kinds["line"][0] / max(1, kinds["line"][1]),
            "attach_rate_gaps": kinds["gap"][0] / max(1, kinds["gap"][1]),
            "collections_on_2plus_patches": multi,
            "same_patch_revisits": len(ss),
            "same_patch_revisits_spread_ge_half_turn": int((ss >= 0.5).sum()),
            "pairs_vs_reference": cmp_,
            "abs_residual_vs_reference_p50_p90": (
                np.percentile(np.abs(residuals), [50, 90]).round(3).tolist() if residuals else None)}
    for cid, c in gen.items():
        step, span, r = angle_stats(c["p"], center)
        on = sorted({ps.names[min(h, key=lambda kk: h[kk][0])] for h in hits[cid]
                     if h and _dmin(h) <= args.tol[0]})
        report["per_collection"].append({"id": cid, "name": c["name"], "n": len(c["p"]),
                                         "max_dtheta_rad": round(step, 4), "span_turns": round(span, 4),
                                         "median_radius": round(r, 1), "patches_at_tol0": on})
    report["max_dtheta_rad_overall"] = max(x["max_dtheta_rad"] for x in report["per_collection"])
    spans = [x["span_turns"] for x in report["per_collection"]]
    report["span_turns_p50_max"] = [float(np.median(spans)), max(spans)]
    print(json.dumps({k: v for k, v in report.items() if k != "per_collection"}, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
