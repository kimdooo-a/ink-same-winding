#!/usr/bin/env python3
"""Same-winding constraints from ink predictions.

Reads a flattened segment's ink prediction (2D image aligned with the segment's
tifxyz grid) and finds two kinds of features that, by construction of the
papyrus, lie on one wrap of the sheet:

  * text lines  - the letters of one line inside one column
  * intercolumnar gaps - the blank vertical band between two text columns

Each feature becomes one point collection: its (u, v) image samples are mapped
through the tifxyz grid (x.tif, y.tif, z.tif) to 3D and written in the
VC3D / spiral-fitting point-collection format (``same_windings.json``):

  {"vc_pointcollections_json_version": "1",
   "collections": {"<id>": {"name", "points": {"<pid>": {"p": [x, y, z],
                    "creation_time", "wind_a": null}}, "metadata", "color"}}}

Collections carry no ``wind_a`` -> spiral-fitting treats them as same-winding
(point_collection.normalise_pcl_winding_annotations). Point ids follow the
feature (left->right for lines, top->bottom for gaps) because unattached
collections are consumed as ordered strips.

The method is deliberately classical (projection profiles + peak linking) so
every constraint can be explained and checked by eye in the debug overlay.

It only re-uses ink predictions that are already published; it does not read
or transcribe text.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.signal import find_peaks

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None

Image.MAX_IMAGE_PIXELS = None
TOOL_NAME = "ink_same_winding"
TOOL_VERSION = "0.2.1"


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------

def load_tifxyz(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (xyz[H, W, 3] float32, valid[H, W] bool, meta)."""
    xyz = np.stack([tifffile.imread(os.path.join(path, f"{c}.tif")).astype(np.float32)
                    for c in "xyz"], axis=-1)
    with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    valid = np.all(xyz > 0, axis=-1) & np.all(np.isfinite(xyz), axis=-1)
    return xyz, valid, meta


def load_ink(path: str) -> np.ndarray:
    """Ink probability image as float32 in [0, 1]."""
    if path.lower().endswith((".tif", ".tiff")):
        a = tifffile.imread(path)
    else:
        a = np.asarray(Image.open(path).convert("L"))
    a = a.astype(np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    hi = 65535.0 if a.max() > 255 else 255.0
    return a / hi


def find_inputs(segment_dir: str, volume_hint: str | None) -> tuple[str, str]:
    """Pick (ink image, tifxyz dir) in a downloaded segment directory.

    Prefers the downsampled ink JPG (enough resolution for line/column
    geometry) and the tifxyz whose grid the ink image is aligned to.
    """
    inks = sorted(glob.glob(os.path.join(glob.escape(segment_dir), "ink-detection", "downsampled", "*.jpg")))
    inks += sorted(glob.glob(os.path.join(glob.escape(segment_dir), "ink-detection", "*.tif")))
    if volume_hint:
        inks = [p for p in inks if volume_hint in os.path.basename(p)] or inks
    meshes = sorted(glob.glob(os.path.join(glob.escape(segment_dir), "mesh", "*-on-*.tifxyz")))
    if volume_hint:
        meshes = [p for p in meshes if volume_hint in os.path.basename(p)] or meshes
    if not inks or not meshes:
        raise SystemExit(f"could not find ink image / tifxyz under {segment_dir}")
    return inks[0], meshes[0]


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def bilinear_xyz(xyz: np.ndarray, valid: np.ndarray, gi: np.ndarray, gj: np.ndarray):
    """Bilinear lookup of grid (row gi, col gj). Returns (pts[N,3], ok[N])."""
    h, w = valid.shape
    i0 = np.floor(gi).astype(np.int64)
    j0 = np.floor(gj).astype(np.int64)
    ok = (i0 >= 0) & (j0 >= 0) & (i0 + 1 < h) & (j0 + 1 < w)
    i0c = np.clip(i0, 0, h - 2)
    j0c = np.clip(j0, 0, w - 2)
    di = (gi - i0c)[:, None]
    dj = (gj - j0c)[:, None]
    c00, c01 = xyz[i0c, j0c], xyz[i0c, j0c + 1]
    c10, c11 = xyz[i0c + 1, j0c], xyz[i0c + 1, j0c + 1]
    ok &= valid[i0c, j0c] & valid[i0c, j0c + 1] & valid[i0c + 1, j0c] & valid[i0c + 1, j0c + 1]
    pts = (c00 * (1 - di) * (1 - dj) + c01 * (1 - di) * dj
           + c10 * di * (1 - dj) + c11 * di * dj)
    return pts, ok


def smooth1d(a: np.ndarray, sigma: float) -> np.ndarray:
    return ndi.gaussian_filter1d(a.astype(np.float64), sigma, mode="nearest")


def masked_mean(values: np.ndarray, mask: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    num = (values * mask).sum(axis=axis)
    den = mask.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1), np.nan), den


# ----------------------------------------------------------------------------
# Feature detection
# ----------------------------------------------------------------------------

@dataclass
class Feature:
    kind: str                 # "line" | "gap"
    uv: np.ndarray            # (N, 2) work-image (u=col, v=row) samples, chain order
    confidence: float
    info: dict = field(default_factory=dict)


@dataclass
class Params:
    px_mm: float              # work-image pixel size in mm
    gap_min_mm: float = 6.0   # minimum gap width
    col_min_mm: float = 25.0  # minimum text column width
    band_mm: float = 12.0     # v band height for gap path refinement
    gap_search_mm: float = 8.0
    strip_mm: float = 12.0    # u sub-strip width for line peaks
    line_sigma_mm: float = 0.6
    sample_mm: float = 1.0    # spacing of emitted points
    min_line_frac: float = 0.5
    ink_presence: float = 0.25  # relative ink level that counts as "letter here"
    min_gap_contrast: float = 0.2
    max_skip_mm: float = 4.0
    gap_prominence: float = 0.15  # x (p90 - p5) of the column profile
    min_line_strips: int = 3      # a line chain spanning >= this many strips is kept  # split a line where no letter is seen for this long
    min_line_conf: float = 0.3

    def px(self, mm: float) -> float:
        return mm / self.px_mm


def detect_gaps(ink: np.ndarray, mask: np.ndarray, P: Params):
    """Intercolumnar gaps as vertical low-ink paths between text columns.

    Returns (gaps: List[Feature], column_bounds: List[(u0, u1)], profile)."""
    H, W = ink.shape
    prof, cover = masked_mean(ink, mask, axis=0)
    good = cover > 0.5 * H * (cover.max() / H)
    prof = np.where(good, prof, np.nan)
    filled = np.where(np.isfinite(prof), prof, np.nanmedian(prof))
    sm = smooth1d(filled, P.px(P.gap_min_mm) / 2)
    hi = np.nanpercentile(sm[good], 90)
    lo = np.nanpercentile(sm[good], 5)
    # gaps = prominent minima, at least one column apart
    peaks, props = find_peaks(-sm, distance=max(1, int(P.px(P.col_min_mm))),
                              prominence=P.gap_prominence * (hi - lo))
    gaps: list[Feature] = []
    u_valid = np.flatnonzero(good)
    u_min, u_max = (u_valid.min(), u_valid.max()) if len(u_valid) else (0, W - 1)
    margin = P.px(P.gap_min_mm)
    band = max(4, int(P.px(P.band_mm)))
    flank_w = int(P.px(P.gap_search_mm))
    for k, u_c in enumerate(peaks):
        if u_c - u_min < margin or u_max - u_c < margin:
            continue  # segment edge, not an intercolumnar gap
        # measured gap half-width: where the smoothed profile climbs half-way
        # back to the neighbouring column level (gaps are 5-20 mm wide)
        reach = int(P.px(P.col_min_mm) * 0.5)
        left = sm[max(0, u_c - reach):u_c + 1][::-1]
        right = sm[u_c:min(W, u_c + reach + 1)]
        half = 0.5 * (sm[u_c] + min(left.max(), right.max()))
        hw_l = int(np.argmax(left >= half)) if (left >= half).any() else len(left)
        hw_r = int(np.argmax(right >= half)) if (right >= half).any() else len(right)
        search = max(int(P.px(P.gap_min_mm) / 2), int(0.75 * max(hw_l, hw_r)))
        path = []
        contrasts = []
        for v0 in range(0, H, band):
            v1 = min(H, v0 + band)
            a, b = max(0, u_c - search), min(W, u_c + search + 1)
            sub = ink[v0:v1, a:b]
            sm_mask = mask[v0:v1, a:b]
            if sm_mask.mean() < 0.6:
                continue
            p, _ = masked_mean(sub, sm_mask, axis=0)
            p = smooth1d(np.nan_to_num(p, nan=np.nanmean(p)), P.px(P.gap_min_mm) / 3)
            j = int(np.argmin(p))
            # flank density: ink in the text columns on either side, beyond
            # the measured gap edges
            fa0, fa1 = max(0, u_c - hw_l - flank_w), max(0, u_c - hw_l)
            fb0, fb1 = min(W, u_c + hw_r), min(W, u_c + hw_r + flank_w)
            fl_a = ink[v0:v1, fa0:fa1][mask[v0:v1, fa0:fa1]]
            fl_b = ink[v0:v1, fb0:fb1][mask[v0:v1, fb0:fb1]]
            if fl_a.size < 10 or fl_b.size < 10:
                continue
            flank = min(fl_a.mean(), fl_b.mean())
            contrast = 1.0 - p[j] / max(flank, 1e-6)
            if contrast < P.min_gap_contrast:
                continue
            path.append((a + j, 0.5 * (v0 + v1)))
            contrasts.append(contrast)
        if len(path) < 3:
            continue
        path = np.asarray(path, dtype=np.float64)
        # robustify: gaps are smooth, so fit u(v) with a quadratic and
        # iteratively drop bands further than half a minimum gap width
        contrasts = np.asarray(contrasts)
        keep = np.ones(len(path), bool)
        for _ in range(3):
            deg = 2 if keep.sum() >= 5 else 1
            coef = np.polyfit(path[keep, 1], path[keep, 0], deg)
            res = np.abs(np.polyval(coef, path[:, 1]) - path[:, 0])
            keep = res < P.px(P.gap_min_mm) * 0.5
            if keep.sum() < 3:
                break
        if keep.sum() < 3:
            continue
        vv = np.arange(path[keep, 1].min(), path[keep, 1].max() + 1e-9, P.px(P.sample_mm))
        path = np.c_[np.polyval(coef, vv), vv]
        contrasts = contrasts[keep]
        uv = path
        gaps.append(Feature("gap", uv, float(np.clip(np.mean(contrasts), 0, 1)),
                            {"u_center_px": float(u_c), "bands": int(keep.sum()),
                             "width_mm": float((hw_l + hw_r) * P.px_mm),
                             "mean_contrast": float(np.mean(contrasts))}))
    # A blank band with no text column between it and the segment edge is the
    # sheet's margin (or a column whose text lies outside the segment), not
    # evidence of an intercolumnar gap -> tag it, the caller decides.
    for g in gaps:
        uc = float(np.median(g.uv[:, 0]))
        outer = min(uc - u_min, u_max - uc) * P.px_mm
        g.info["edge_margin"] = bool(outer < 0.8 * P.col_min_mm)
        g.info["outer_mm"] = float(outer)
    # columns between gaps (and the segment's valid u-range)
    cuts = [u_min] + [float(np.median(g.uv[:, 0])) for g in gaps] + [u_max]
    columns = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)
               if cuts[i + 1] - cuts[i] > P.px(P.col_min_mm) * 0.5]
    return gaps, columns, sm


def line_spacing_px(profile: np.ndarray, lo: float, hi: float) -> float | None:
    """Dominant period of a row profile via autocorrelation, in [lo, hi] px."""
    p = profile - np.nanmean(profile)
    p = np.nan_to_num(p)
    ac = np.correlate(p, p, mode="full")[len(p) - 1:]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    a, b = int(lo), min(len(ac) - 1, int(hi))
    if b <= a:
        return None
    k = a + int(np.argmax(ac[a:b]))
    return float(k) if ac[k] > 0.05 else None


def detect_lines(ink: np.ndarray, mask: np.ndarray, columns, gaps, P: Params) -> list[Feature]:
    """Text lines per column: row-profile peaks in narrow u strips, linked
    across strips into near-horizontal chains."""
    H, W = ink.shape
    lines: list[Feature] = []
    sigma = P.px(P.line_sigma_mm)
    for ci, (u0, u1) in enumerate(columns):
        # trim column edges away from the gaps (keep the letters only)
        pad = P.px(P.gap_min_mm) * 0.5
        a, b = int(u0 + pad), int(u1 - pad)
        if b - a < P.px(P.col_min_mm) * 0.5:
            continue
        colprof, _ = masked_mean(ink[:, a:b], mask[:, a:b], axis=1)
        sp = line_spacing_px(smooth1d(np.nan_to_num(colprof, nan=0.0), sigma),
                             P.px(2.0), P.px(9.0))
        if sp is None:
            continue
        nstrip = max(2, int(round((b - a) / P.px(P.strip_mm))))
        edges = np.linspace(a, b, nstrip + 1).astype(int)
        strip_peaks = []
        for s in range(nstrip):
            sa, sb = edges[s], edges[s + 1]
            pr, cov = masked_mean(ink[:, sa:sb], mask[:, sa:sb], axis=1)
            okrow = cov > 0.8 * (sb - sa)
            pr = smooth1d(np.where(okrow, np.nan_to_num(pr), 0.0), sigma)
            base = ndi.minimum_filter1d(pr, int(sp * 1.5), mode="nearest")
            base = smooth1d(base, sp / 2)
            rel = pr - base
            scale = np.percentile(rel[okrow], 95) if okrow.any() else 0
            if scale <= 0:
                strip_peaks.append((0.5 * (sa + sb), np.array([]), np.array([])))
                continue
            pk, pp = find_peaks(rel, distance=max(1, int(0.6 * sp)), prominence=0.3 * scale)
            pk = pk[okrow[pk]]
            prom = rel[pk] / scale
            strip_peaks.append((0.5 * (sa + sb), pk.astype(float), prom))
        # link peaks strip-to-strip (greedy nearest within 0.35 spacing)
        chains: list[list[tuple[float, float, float]]] = []
        open_chains: list[int] = []
        for s, (uc, pk, prom) in enumerate(strip_peaks):
            used = set()
            new_open = []
            for ch_idx in open_chains:
                last = chains[ch_idx][-1]
                if len(pk) == 0:
                    continue
                d = np.abs(pk - last[1])
                j = int(np.argmin(d))
                if d[j] < 0.35 * sp and j not in used:
                    used.add(j)
                    chains[ch_idx].append((uc, pk[j], prom[j]))
                    new_open.append(ch_idx)
            for j in range(len(pk)):
                if j not in used:
                    chains.append([(uc, pk[j], prom[j])])
                    new_open.append(len(chains) - 1)
            open_chains = new_open
        for ch in chains:
            frac = len(ch) / nstrip
            # keep a chain that covers half the column, or at least a few
            # strips (columns merged by a missed gap make "frac" meaningless)
            if len(ch) < 2 or (frac < P.min_line_frac and len(ch) < P.min_line_strips):
                continue
            ch = np.asarray(ch)
            # straightness: residual of a linear fit relative to spacing
            A = np.c_[ch[:, 0], np.ones(len(ch))]
            coef, *_ = np.linalg.lstsq(A, ch[:, 1], rcond=None)
            resid = float(np.sqrt(np.mean((A @ coef - ch[:, 1]) ** 2)))
            straight = max(0.0, 1.0 - resid / (0.25 * sp))
            span = min(1.0, len(ch) * P.strip_mm / 48.0)  # ~ one column of text
            conf = float(np.clip(np.mean(np.minimum(ch[:, 2], 1.0)) * span * straight, 0, 1))
            if conf < P.min_line_conf:
                continue
            # dense samples along the line, kept only where letters are present
            # only where the chain was actually seen (no extrapolation)
            half_strip = 0.5 * (edges[1] - edges[0])
            us = np.arange(max(edges[0], ch[:, 0].min() - half_strip),
                           min(edges[-1], ch[:, 0].max() + half_strip), P.px(P.sample_mm))
            vs = np.interp(us, ch[:, 0], ch[:, 1])
            win_v = max(1, int(0.25 * sp))
            win_u = max(1, int(P.px(P.sample_mm)))
            keep = []
            ref = np.percentile(ink[mask], 99) if mask.any() else 1.0
            for u, v in zip(us, vs):
                iu, iv = int(u), int(round(v))
                patch = ink[max(0, iv - win_v):iv + win_v + 1, max(0, iu - win_u):iu + win_u + 1]
                pm = mask[max(0, iv - win_v):iv + win_v + 1, max(0, iu - win_u):iu + win_u + 1]
                keep.append(pm.all() and patch.max() > P.ink_presence * ref)
            keep = np.asarray(keep)
            if keep.sum() < 3:
                continue
            uv = np.c_[us[keep], vs[keep]]
            # split where no letters were found for a while: spiral-fitting
            # reads unattached collections as dense ordered strips
            # (|dtheta| < pi between neighbours), so do not bridge long blanks
            cut = np.flatnonzero(np.diff(uv[:, 0]) > P.px(P.max_skip_mm)) + 1
            for part, run in enumerate(np.split(uv, cut)):
                if len(run) < 3:
                    continue
                lines.append(Feature("line", run, conf, {
                    "column": ci, "part": part, "line_spacing_px": sp, "strips_hit": len(ch),
                    "strips": int(nstrip), "fit_resid_px": resid,
                    "slope": float(coef[0])}))
    return lines


def resample_polyline(path: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] <= 0:
        return path[:1]
    t = np.arange(0, s[-1] + 1e-9, step)
    return np.c_[np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1])]


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

def hsv_color(k: int) -> list[float]:
    import colorsys
    return list(colorsys.hsv_to_rgb((k * 0.61803398875) % 1.0, 0.8, 1.0))


def evidence_of(f: Feature) -> str:
    if f.kind == "line":
        return "ink_text_line"
    return "edge_margin" if f.info.get("edge_margin") else "intercolumnar_gap"


def features_to_3d(features: list[Feature], xyz, valid, work_to_grid, coord_scale):
    """[(feature, pts[N,3] in output frame, uv_grid[N,2] (col,row))] + #invalid."""
    out, dropped = [], 0
    for f in features:
        gj = f.uv[:, 0] * work_to_grid[1]
        gi = f.uv[:, 1] * work_to_grid[0]
        pts, ok = bilinear_xyz(xyz, valid, gi, gj)
        dropped += int((~ok).sum())
        out.append((f, pts[ok] * coord_scale, np.c_[gj[ok], gi[ok]]))
    return out, dropped


def split_by_turns(records, center, max_turns: float):
    """Cut each chain wherever its unwrapped angle around the umbilicus has
    advanced by ``max_turns`` (same-winding annotations in the public
    spiral-input span p50 0.035 / max 0.49 turns)."""
    out = []
    for f, pts, uvg in records:
        if len(pts) < 2:
            out.append((f, pts, uvg, 0))
            continue
        cx, cy = center(pts[:, 2])
        th = np.unwrap(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
        travel = np.r_[0, np.cumsum(np.abs(np.diff(th)))] / (2 * np.pi)
        chunk = np.floor(travel / max_turns).astype(int)
        for c in np.unique(chunk):
            m = chunk == c
            out.append((f, pts[m], uvg[m], int(c)))
    return out


def build_collections(records, segment_id: str, provenance: dict, min_points: int,
                      first_id: int = 0):
    """records: [(feature, pts, uv_grid, chunk)] -> point collections."""
    collections = {}
    stats = {"records": len(records), "kept": 0, "points": 0}
    now_ms = int(time.time() * 1000)
    cid = first_id
    pid = 0
    for k, (f, pts, uvg, chunk) in enumerate(records):
        if len(pts) < min_points:
            continue
        points = {}
        for p in pts:
            points[str(pid)] = {"p": [float(p[0]), float(p[1]), float(p[2])],
                                "creation_time": now_ms, "wind_a": None}
            pid += 1
        ev = evidence_of(f)
        tag = {"ink_text_line": f"line_c{f.info.get('column')}", "intercolumnar_gap": "gap",
               "edge_margin": "edgemargin"}[ev] + f"_{k:03d}"
        collections[str(cid)] = {
            "name": f"ink_{tag}_{segment_id}",
            "points": points,
            "metadata": {
                "winding_is_absolute": False,
                "generator": TOOL_NAME,
                "generator_version": TOOL_VERSION,
                "evidence": ev,
                "chunk": chunk,
                "segment_id": segment_id,
                "confidence": round(f.confidence, 4),
                # Same keys as VC3D's fiber H/V tag (spiral_helpers.classify_fiber_hv);
                # spiral-fitting reads it only for fibers, so it is informative here.
                "hv_classification": {"automatic_tag": "H" if f.kind == "line" else "V",
                                      "automatic_certainty": round(f.confidence, 4)},
                "uv_grid_first_last": [[float(uvg[0, 0]), float(uvg[0, 1])],
                                       [float(uvg[-1, 0]), float(uvg[-1, 1])]],
                "detail": {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in f.info.items()},
                **provenance,
            },
            "color": hsv_color(k),
        }
        cid += 1
        stats["kept"] += 1
        stats["points"] += len(points)
    return collections, stats


def write_debug_png(path: str, ink: np.ndarray, mask: np.ndarray, gaps, lines, max_w: int = 4000):
    H, W = ink.shape
    s = min(1.0, max_w / W)
    base = (np.clip(ink / max(np.percentile(ink[mask], 99.5), 1e-6), 0, 1) * 200).astype(np.uint8)
    base[~mask] = 30
    im = Image.fromarray(base).convert("RGB")
    if s < 1.0:
        im = im.resize((int(W * s), int(H * s)), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    for g in gaps:
        pts = [(u * s, v * s) for u, v in g.uv]
        d.line(pts, fill=(255, 60, 60), width=3)
    for i, ln in enumerate(lines):
        col = tuple(int(255 * c) for c in hsv_color(i))
        pts = [(u * s, v * s) for u, v in ln.uv]
        for (u, v) in pts:
            d.ellipse([u - 1.5, v - 1.5, u + 1.5, v + 1.5], fill=col)
    im.save(path)


def _validator():
    """Sibling module (python -I drops the script dir from sys.path)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    return __import__("validate_against_patches")


def filter_isolated_patch_hits(records, args, pv):
    """Drop points whose nearest verified patch (within --patch-tol) is touched
    by fewer than --min-patch-pts points of the same collection."""
    allp = np.concatenate([r[1] for r in records if len(r[1])])
    names, lo, hi = pv.patch_bboxes(args.patch_meta_dir)
    t = args.patch_tol
    sel = [n for n, l, h in zip(names, lo, hi)
           if np.any(np.all((allp >= l - t) & (allp <= h + t), axis=1))]
    ps = pv.PatchSet({n: pv.fetch_patch(n, args.patch_root, args.patch_cache) for n in sel})
    out, dropped, kept_attached = [], 0, 0
    for f, pts, uvg, chunk in records:
        hits = ps.nearest(pts, t)
        near = [min(h, key=lambda kk: h[kk][0]) if h else None for h in hits]
        counts = {}
        for k in near:
            if k is not None:
                counts[k] = counts.get(k, 0) + 1
        keep = np.array([k is None or counts[k] >= args.min_patch_pts for k in near], bool)
        dropped += int((~keep).sum())
        kept_attached += int(sum(1 for k, kp in zip(near, keep) if k is not None and kp))
        out.append((f, pts[keep], uvg[keep], chunk))
    return out, {"candidate_patches": len(sel), "points_dropped": dropped,
                 "points_kept_on_patches": kept_attached}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None):
    for stream in (sys.stdout, sys.stderr):  # paths may contain non-ASCII
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    ap =argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("segment_dir", help="segment directory (contains ink-detection/ and mesh/)")
    ap.add_argument("-o", "--out", required=True, help="output same_windings-style json")
    ap.add_argument("--ink", help="ink image (default: auto-detect under segment_dir)")
    ap.add_argument("--tifxyz", help="tifxyz dir the ink image is aligned to (default: auto)")
    ap.add_argument("--volume", default=None, help="volume id hint for auto-detection")
    ap.add_argument("--ink-px-um", type=float, default=None,
                    help="ink image pixel size in um (default: derived from grid step)")
    ap.add_argument("--coord-scale", type=float, default=0.25,
                    help="multiply tifxyz coords by this (default 0.25: level-0 2.4um -> "
                         "level-2 9.6um, the spiral-input frame)")
    ap.add_argument("--voxel-um", type=float, default=2.4,
                    help="voxel size (um) of the tifxyz coordinates (default 2.4)")
    ap.add_argument("--work-downsample", type=int, default=2,
                    help="extra integer downsample of the ink image before detection")
    ap.add_argument("--segment-id", default=None)
    ap.add_argument("--no-lines", action="store_true")
    ap.add_argument("--no-gaps", action="store_true")
    ap.add_argument("--min-points", type=int, default=5)
    ap.add_argument("--sample-mm", type=float, default=1.0)
    ap.add_argument("--debug-png", default=None)
    ap.add_argument("--keep-edge-margins", action="store_true",
                    help="also emit blank bands next to the segment edge (evidence=edge_margin)")
    ap.add_argument("--umbilicus", default=None,
                    help="umbilicus.json (needed by --max-turns and --min-patch-pts)")
    ap.add_argument("--max-turns", type=float, default=0.0,
                    help="cut chains every N turns around the umbilicus (e.g. 0.25); 0 = off")
    ap.add_argument("--min-patch-pts", type=int, default=0,
                    help="drop points that attach to a verified patch which the same collection "
                         "touches with fewer than N points (isolated hits are mostly the "
                         "neighbouring winding); 0 = off")
    ap.add_argument("--patch-meta-dir", help="dir of <patch>.json meta files (bbox index)")
    ap.add_argument("--patch-root", help="URL or dir holding <patch>/{x,y,z}.tif")
    ap.add_argument("--patch-cache", help="local cache dir for patch tifs")
    ap.add_argument("--patch-tol", type=float, default=2.5,
                    help="attachment distance (spiral-fitting default pcl_link_distance_tolerance)")
    args = ap.parse_args(argv)
    if (args.max_turns > 0 or args.min_patch_pts > 0) and not args.umbilicus:
        ap.error("--max-turns / --min-patch-pts need --umbilicus")
    if args.min_patch_pts > 0 and not (args.patch_meta_dir and args.patch_root and args.patch_cache):
        ap.error("--min-patch-pts needs --patch-meta-dir, --patch-root and --patch-cache")

    ink_path, mesh_path = find_inputs(args.segment_dir, args.volume)
    ink_path = args.ink or ink_path
    mesh_path = args.tifxyz or mesh_path
    segment_id = args.segment_id or os.path.basename(os.path.normpath(args.segment_dir))
    print(f"ink    : {ink_path}\ntifxyz : {mesh_path}")

    xyz, valid, meta = load_tifxyz(mesh_path)
    ink = load_ink(ink_path)
    gh, gw = valid.shape
    ih, iw = ink.shape
    rh, rw = ih / gh, iw / gw
    if abs(rh - rw) > 1e-3 * rh:
        raise SystemExit(f"ink {ink.shape} and grid {valid.shape} have different aspect ratios "
                         f"({rh:.4f} vs {rw:.4f}); pass the tifxyz the ink was rendered from")
    # grid step in voxels (measured, not assumed)
    du = np.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1)[valid[:, 1:] & valid[:, :-1]]
    dv = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)[valid[1:] & valid[:-1]]
    step_vox = float(np.median(np.r_[du, dv]))
    ink_px_um = args.ink_px_um or step_vox * args.voxel_um / rh
    ds = max(1, args.work_downsample)
    if ds > 1:
        h2, w2 = ih // ds, iw // ds
        ink = ink[:h2 * ds, :w2 * ds].reshape(h2, ds, w2, ds).mean(axis=(1, 3))
    work_px_mm = ink_px_um * ds / 1000.0
    work_to_grid = (ds / rh, ds / rw)
    # validity mask at work resolution (nearest), eroded 1 mm from the edge
    vi = (np.arange(ink.shape[0]) * work_to_grid[0]).astype(int).clip(0, gh - 1)
    vj = (np.arange(ink.shape[1]) * work_to_grid[1]).astype(int).clip(0, gw - 1)
    mask = valid[np.ix_(vi, vj)]
    mask = ndi.binary_erosion(mask, iterations=max(1, int(1.0 / work_px_mm)))
    print(f"grid {valid.shape} step {step_vox:.2f} vox | ink {ih}x{iw} ({rh:.3f} px/cell) "
          f"| work {ink.shape} @ {work_px_mm * 1000:.1f} um/px")

    P = Params(px_mm=work_px_mm, sample_mm=args.sample_mm)
    gaps, columns, _ = detect_gaps(ink, mask, P)
    lines = [] if args.no_lines else detect_lines(ink, mask, columns, gaps, P)
    if args.no_gaps:
        gaps = []
    n_edge = sum(1 for g in gaps if g.info.get("edge_margin"))
    if not args.keep_edge_margins:
        gaps = [g for g in gaps if not g.info.get("edge_margin")]
    print(f"columns {len(columns)} | gaps {len(gaps)} (edge margins found {n_edge}, "
          f"{'kept' if args.keep_edge_margins else 'dropped'}) | lines {len(lines)}")

    provenance = {"ink_file": os.path.basename(ink_path),
                  "tifxyz": os.path.basename(os.path.normpath(mesh_path)),
                  "coord_scale": args.coord_scale,
                  "coord_order": "xyz"}
    records, dropped_invalid = features_to_3d(gaps + lines, xyz, valid, work_to_grid,
                                              args.coord_scale)
    center = None
    if args.umbilicus:
        center = _validator().umbilicus_center(args.umbilicus)
        provenance["umbilicus"] = os.path.basename(args.umbilicus)
    if args.max_turns > 0:
        records = split_by_turns(records, center, args.max_turns)
        provenance["max_turns"] = args.max_turns
    else:
        records = [(f, p, u, 0) for f, p, u in records]
    filt = None
    if args.min_patch_pts > 0:
        records, filt = filter_isolated_patch_hits(records, args, _validator())
        provenance["min_patch_pts"] = args.min_patch_pts
        provenance["patch_tol"] = args.patch_tol
    collections, stats = build_collections(records, segment_id, provenance, args.min_points)
    stats["dropped_invalid"] = dropped_invalid
    if filt:
        stats["patch_filter"] = filt
    out = {"vc_pointcollections_json_version": "1", "collections": collections}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"wrote {args.out}: {stats}")
    if args.debug_png:
        write_debug_png(args.debug_png, ink, mask, gaps, lines)
        print(f"debug overlay: {args.debug_png}")
    return out, stats


if __name__ == "__main__":
    main()
