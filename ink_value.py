#!/usr/bin/env python3
"""What does the ink add beyond the segment itself?

Two measurements on one segment, both against public data only.

(a) Ink coherence vs. segment winding errors.
    Every verified-patch vertex that lies on the segment surface (normal
    distance <= --tol, tangential <= one grid step) gives an offset
    o = W_segment(cell) - W_patch(vertex), W = unwrapped angle around the
    umbilicus / 2pi (W_segment unwrapped along the segment grid, W_patch along
    the patch grid). Patches that lie on the segment over >= --anchor-turns
    are anchors. In bins of the segment's own winding coordinate, a whole-turn
    change of o shared by most anchors is a segment winding error (sheet
    skip); a change seen by a minority is a patch problem. We then compare the
    ink-line coverage of text cells in error bins vs. the other bins.

(b) Coverage. Patch pairs that the generated collections connect, compared
    with the graph of pairs the published annotations already connect
    (directly or through other patches).

usage:
  ink_value.py SEGMENT_DIR GENERATED.json --patch-meta-dir D --patch-root R --patch-cache C
      --umbilicus U --relative REL --same SAME [--tol 2.5] [--out report.json] [--map map.png]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # siblings under python -I
isw = __import__("ink_same_winding")
v = __import__("validate_against_patches")


def segment_w(xyz, valid, center):
    """W along the segment grid: rows unwrapped along u, then rows aligned to
    each other by whole turns (the grid is one connected sheet)."""
    cx, cy = center(np.where(valid, xyz[..., 2], np.nan))
    th = np.arctan2(xyz[..., 1] - cy, xyz[..., 0] - cx)
    # fill holes from the nearest valid cell so np.unwrap sees a continuous row
    idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    thf = th[tuple(idx)]
    un = np.unwrap(thf, axis=1)
    for i in range(1, un.shape[0]):
        k = np.round(np.median(un[i] - un[i - 1]) / (2 * np.pi))
        un[i] -= 2 * np.pi * k
    w = un / (2 * np.pi)
    w[~valid] = np.nan
    return w


def grid_normals(xyz, valid):
    du = np.gradient(xyz, axis=1)
    dv = np.gradient(xyz, axis=0)
    n = np.cross(du, dv)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    n[~valid] = 0
    return n


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb
            return True
        return False


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("segment_dir")
    ap.add_argument("generated")
    ap.add_argument("--patch-meta-dir", required=True)
    ap.add_argument("--patch-root", required=True)
    ap.add_argument("--patch-cache", required=True)
    ap.add_argument("--umbilicus", required=True)
    ap.add_argument("--relative", required=True)
    ap.add_argument("--same", required=True)
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--stride", type=int, default=2, help="segment grid subsampling")
    ap.add_argument("--coord-scale", type=float, default=0.25)
    ap.add_argument("--min-patch-pts", type=int, default=3)
    ap.add_argument("--bin-turns", type=float, default=0.05,
                    help="bin size along the segment's winding coordinate")
    ap.add_argument("--anchor-min-vertices", type=int, default=5000)
    ap.add_argument("--anchor-turns", type=float, default=0.5,
                    help="an anchor patch must lie on the segment over at least this many turns")
    ap.add_argument("--out")
    ap.add_argument("--map", help="PNG: text cells with a detected line (blue), bins where the "
                                  "segment shifts winding (red), bins where one patch shifts (amber)")
    args = ap.parse_args()

    # ---- segment geometry
    ink_path, mesh_path = isw.find_inputs(args.segment_dir, None)
    xyz0, valid0, _ = isw.load_tifxyz(mesh_path)
    s = args.stride
    xyz = xyz0[::s, ::s] * args.coord_scale
    valid = valid0[::s, ::s]
    center = v.umbilicus_center(args.umbilicus)
    wseg = segment_w(xyz, valid, center)
    nrm = grid_normals(xyz, valid)
    ii, jj = np.nonzero(valid)
    tree = cKDTree(xyz[ii, jj])
    spacing = float(np.median(np.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1)[valid[:, 1:] & valid[:, :-1]]))
    print(f"segment grid {valid.shape} (stride {s}) spacing {spacing:.2f} vox, "
          f"W span {np.nanmax(wseg) - np.nanmin(wseg):.2f} turns", flush=True)

    # ---- ink coherence map on the same grid (re-run the detector)
    ink = isw.load_ink(ink_path)
    gh, gw = valid0.shape
    rh = ink.shape[0] / gh
    ds = 2
    h2, w2 = ink.shape[0] // ds, ink.shape[1] // ds
    ink = ink[:h2 * ds, :w2 * ds].reshape(h2, ds, w2, ds).mean(axis=(1, 3))
    du = np.linalg.norm(xyz0[:, 1:] - xyz0[:, :-1], axis=-1)[valid0[:, 1:] & valid0[:, :-1]]
    ink_px_um = float(np.median(du)) * 2.4 / rh
    P = isw.Params(px_mm=ink_px_um * ds / 1000.0)
    w2g = ds / rh
    vi = (np.arange(ink.shape[0]) * w2g).astype(int).clip(0, gh - 1)
    vj = (np.arange(ink.shape[1]) * w2g).astype(int).clip(0, gw - 1)
    mask = ndi.binary_erosion(valid0[np.ix_(vi, vj)], iterations=max(1, int(1.0 / P.px_mm)))
    gaps, columns, _ = isw.detect_gaps(ink, mask, P)
    lines = isw.detect_lines(ink, mask, columns, gaps, P)
    in_col = np.zeros(valid.shape, bool)
    for u0, u1 in columns:
        a, b = int(u0 * w2g / s), int(u1 * w2g / s)
        in_col[:, a:b] = True
    covered = np.zeros(valid.shape, bool)
    for ln in lines:
        gi = (ln.uv[:, 1] * w2g / s).astype(int).clip(0, valid.shape[0] - 1)
        gj = (ln.uv[:, 0] * w2g / s).astype(int).clip(0, valid.shape[1] - 1)
        covered[gi, gj] = True
    sp_cells = np.median([ln.info["line_spacing_px"] for ln in lines]) * w2g / s if lines else 10
    # a cell is "ink-coherent" if a detected line passes within half a line
    # spacing vertically and one sample horizontally
    covered = ndi.binary_dilation(covered, structure=np.ones((max(1, int(sp_cells)), max(1, int(2 * P.px(1.0) * w2g / s))), bool))
    covered &= in_col & valid
    print(f"ink: {len(columns)} columns, {len(lines)} lines; coherent cells "
          f"{covered.sum() / max(1, (in_col & valid).sum()):.1%} of text-column cells", flush=True)

    # ---- patches on the segment
    allp = xyz[ii, jj]
    names, lo, hi = v.patch_bboxes(args.patch_meta_dir)
    # keep patches whose (expanded) bbox actually contains segment points
    m_ = args.tol + spacing
    sample = allp[::7]
    coarse = cKDTree(sample)
    keep = []
    for n, l, h in zip(names, lo, hi):
        c = 0.5 * (l + h)
        idx = coarse.query_ball_point(c, r=0.5 * np.linalg.norm(h - l) + m_)
        if idx:
            q = sample[idx]
            if np.any(np.all((q >= l - m_) & (q <= h + m_), axis=1)):
                keep.append(n)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as ex:
        grids = dict(zip(keep, ex.map(lambda n: v.fetch_patch(n, args.patch_root, args.patch_cache), keep)))
    ps = v.PatchSet(grids)
    wf = v.WindingField(ps, center)
    print(f"patches near segment: {len(keep)}", flush=True)

    # vertex -> segment cell (normal distance <= tol, tangential <= spacing)
    d, nn = tree.query(ps.verts, distance_upper_bound=args.tol + spacing)
    ok = np.isfinite(d)
    ci, cj = ii[nn[ok]], jj[nn[ok]]
    vec = ps.verts[ok] - xyz[ci, cj]
    nd = np.abs(np.einsum("ij,ij->i", vec, nrm[ci, cj]))
    tg = np.linalg.norm(vec - (vec * nrm[ci, cj]).sum(1, keepdims=True) * nrm[ci, cj], axis=1)
    on = (nd <= args.tol) & (tg <= spacing)
    vid = np.flatnonzero(ok)[on]
    ci, cj = ci[on], cj[on]
    owner = ps.owner[vid]
    wp = np.array([wf.patch_w(int(k), int(a), int(b)) for k, (a, b) in zip(owner, ps.ij[vid])])
    off = wseg[ci, cj] - wp
    good = np.isfinite(off)
    vid, ci, cj, owner, off = vid[good], ci[good], cj[good], owner[good], off[good]
    print(f"patch vertices on the segment surface: {len(vid)} from {len(np.unique(owner))} patches", flush=True)

    # ---- (a) segment winding errors, located with large verified patches
    # An anchor is a verified patch that lies on the segment over >= --anchor-turns
    # of the segment's own winding coordinate. Along a correct segment the
    # offset W_segment - W_anchor is one constant. A sheet skip in the segment
    # shifts it by a whole turn for every anchor at once; a shift seen by one
    # anchor only is that patch's problem, not the segment's.
    bins = np.floor((wseg - np.nanmin(wseg)) / args.bin_turns)
    vbin = bins[ci, cj].astype(int)
    nbin = int(np.nanmax(bins)) + 1
    anchors = []
    for k in np.unique(owner):
        m = owner == k
        if m.sum() < args.anchor_min_vertices:
            continue
        span = float(np.ptp(wseg[ci[m], cj[m]]))
        if span < args.anchor_turns:
            continue
        anchors.append(int(k))
    shift = np.full((len(anchors), nbin), np.nan)
    anchor_rows = []
    for r, k in enumerate(anchors):
        m = owner == k
        o = off[m]
        maj = float(np.median(o))
        for bb in np.unique(vbin[m]):
            sel = vbin[m] == bb
            if sel.sum() >= 20:
                shift[r, bb] = np.round(np.median(o[sel]) - maj)
        cov = np.isfinite(shift[r])
        anchor_rows.append({"patch": ps.names[k], "on_surface_vertices": int(m.sum()),
                            "span_turns": round(float(np.ptp(wseg[ci[m], cj[m]])), 3),
                            "bins_covered": int(cov.sum()),
                            "bins_shifted": int((cov & (shift[r] != 0)).sum()),
                            "vertex_agreement": round(float(np.mean(np.abs(o - maj) < 0.5)), 4)})
    anchor_rows.sort(key=lambda x: -x["on_surface_vertices"])
    seg_err, patch_err, judged_bins = [], [], []
    for bb in range(nbin):
        col = shift[:, bb]
        col = col[np.isfinite(col)]
        if len(col) < 2:
            continue
        judged_bins.append(bb)
        frac = float(np.mean(col != 0))
        if frac > 0.5:
            seg_err.append(bb)        # most anchors shifted together
        elif frac > 0:
            patch_err.append(bb)      # a minority of anchors shifted
    # ink coherence per bin (text-column cells covered by a detected line)
    text = in_col & valid
    coh = {}
    for bb in judged_bins:
        cells = text & (bins == bb)
        if cells.sum():
            coh[bb] = float((covered & cells).sum() / cells.sum())
    def mean_coh(bs):
        vals = [coh[x] for x in bs if x in coh]
        return (float(np.mean(vals)) if vals else None, len(vals))
    ok_bins = [x for x in judged_bins if x not in seg_err]
    a_result = {
        "anchors": anchor_rows,
        "bin_turns": args.bin_turns,
        "bins_judged_by_2plus_anchors": len(judged_bins),
        "bins_segment_shifted": len(seg_err),
        "bins_single_patch_shifted": len(patch_err),
        "ink_coherence_in_ok_bins": mean_coh(ok_bins),
        "ink_coherence_in_segment_error_bins": mean_coh(seg_err),
    }
    if seg_err:
        # detector: "ink coherence below the segment's median" -> error bin
        thr = float(np.median(list(coh.values())))
        pred = {x for x in coh if coh[x] < thr}
        truth = set(seg_err) & set(coh)
        tp = len(pred & truth)
        a_result["detector_low_coherence"] = {
            "threshold": thr, "precision": tp / max(1, len(pred)),
            "recall": tp / max(1, len(truth)), "tp": tp,
            "fp": len(pred - truth), "fn": len(truth - pred)}
    else:
        a_result["detector_low_coherence"] = ("not measurable: no bin where most anchors shift "
                                              "together (no segment winding error found)")

    # reference relations (published annotations), kept for coverage (b)
    ref = {}
    for path in (args.relative, args.same):
        for c in v.load_pcl(path).values():
            hits = ps.nearest(c["p"], args.tol)
            for key, val in v.pair_claims(v.patch_offsets(c, hits, wf, center, args.tol)).items():
                ref.setdefault(key, []).append(val)
    ref = {k: float(np.median(x)) for k, x in ref.items()}

    # ---- (b) coverage
    gen = v.load_pcl(args.generated)
    ref_uf = UnionFind()
    for (a, b) in ref:
        ref_uf.union(a, b)
    ours_pairs = set()
    for c in gen.values():
        hits = ps.nearest(c["p"], args.tol)
        offs = v.patch_offsets(c, hits, wf, center, args.tol)
        offs = {k: x for k, x in offs.items() if len(x) >= args.min_patch_pts}
        ks = sorted(offs)
        ours_pairs |= {(a, b) for i, a in enumerate(ks) for b in ks[i + 1:]}
    touched = {k for pr in ours_pairs for k in pr}
    new_direct = [pr for pr in ours_pairs if pr not in ref]
    new_component = [pr for pr in new_direct if ref_uf.find(pr[0]) != ref_uf.find(pr[1])]
    comps_before = len({ref_uf.find(k) for k in touched})
    uf2 = UnionFind()
    for k in touched:
        uf2.union(ref_uf.find(k), ref_uf.find(k))
    for a, b in ours_pairs:
        uf2.union(ref_uf.find(a), ref_uf.find(b))
    comps_after = len({uf2.find(ref_uf.find(k)) for k in touched})
    no_ref = [k for k in touched if not any(k in pr for pr in ref)]
    coverage = {"patch_pairs_ours": len(ours_pairs), "patches_touched": len(touched),
                "pairs_already_in_reference": len(ours_pairs) - len(new_direct),
                "pairs_new_direct": len(new_direct),
                "pairs_joining_reference_components": len(new_component),
                "reference_components_among_touched_before": comps_before,
                "after_adding_ours": comps_after,
                "touched_patches_with_no_reference_relation": len(no_ref)}

    report = {"segment": os.path.basename(os.path.normpath(args.segment_dir)),
              "grid_stride": s, "tol": args.tol,
              "segment_W_span_turns": float(np.nanmax(wseg) - np.nanmin(wseg)),
              "ink": {"columns": len(columns), "lines": len(lines),
                      "coherent_fraction_of_text_cells": float(covered.sum() / max(1, text.sum()))},
              "patches_on_segment": int(len(np.unique(owner))),
              "vertices_on_segment": int(len(vid)),
              "a_segment_errors": a_result,
              "b_coverage": coverage}
    print(json.dumps(report, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    if args.map:
        img = np.zeros(valid.shape + (3,), np.uint8)
        img[valid] = (40, 40, 40)
        img[covered] = (90, 90, 140)
        for bb in seg_err:
            img[(bins == bb) & valid] = (200, 40, 40)
        for bb in patch_err:
            img[(bins == bb) & valid & ~covered] = (150, 110, 30)
        im = Image.fromarray(img)
        w_ = min(4000, im.width)
        im.resize((w_, int(im.height * w_ / im.width)), Image.NEAREST).save(args.map)


if __name__ == "__main__":
    main()
