#!/usr/bin/env python3
"""Control collections with the same geometry budget as the ink lines.

For every detected text line, emit a chain over the same u range on the same
segment, but
  --mode interline : half a line spacing lower (blank papyrus between lines)
  --mode random    : at a uniformly random row of the segment
Chunking (--max-turns) is the same as for the real output. Validating these
with validate_against_patches.py separates what the segment geometry gives
(any chain on the segment) from what the ink adds.

usage: make_controls.py SEGMENT_DIR -o OUT.json --mode {interline,random}
       --umbilicus U [--max-turns 0.25] [--seed 0]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # siblings under python -I
isw = __import__("ink_same_winding")
v = __import__("validate_against_patches")


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("segment_dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--mode", choices=["interline", "random"], required=True)
    ap.add_argument("--umbilicus", required=True)
    ap.add_argument("--max-turns", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coord-scale", type=float, default=0.25)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    ink_path, mesh_path = isw.find_inputs(args.segment_dir, None)
    xyz, valid, _ = isw.load_tifxyz(mesh_path)
    ink = isw.load_ink(ink_path)
    gh, gw = valid.shape
    rh = ink.shape[0] / gh
    ds = 2
    h2, w2 = ink.shape[0] // ds, ink.shape[1] // ds
    ink = ink[:h2 * ds, :w2 * ds].reshape(h2, ds, w2, ds).mean(axis=(1, 3))
    du = np.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1)[valid[:, 1:] & valid[:, :-1]]
    P = isw.Params(px_mm=float(np.median(du)) * 2.4 / rh * ds / 1000.0)
    w2g = ds / rh
    vi = (np.arange(ink.shape[0]) * w2g).astype(int).clip(0, gh - 1)
    vj = (np.arange(ink.shape[1]) * w2g).astype(int).clip(0, gw - 1)
    mask = ndi.binary_erosion(valid[np.ix_(vi, vj)], iterations=max(1, int(1.0 / P.px_mm)))
    gaps, columns, _ = isw.detect_gaps(ink, mask, P)
    lines = isw.detect_lines(ink, mask, columns, gaps, P)

    controls = []
    for ln in lines:
        uv = ln.uv.copy()
        if args.mode == "interline":
            uv[:, 1] += 0.5 * ln.info["line_spacing_px"]
        else:
            # random row; keep the line's shape, require the row to be on the mask
            for _ in range(50):
                dv = rng.uniform(0, mask.shape[0]) - uv[:, 1].mean()
                cand = uv[:, 1] + dv
                iv = np.clip(cand.astype(int), 0, mask.shape[0] - 1)
                iu = np.clip(uv[:, 0].astype(int), 0, mask.shape[1] - 1)
                if mask[iv, iu].mean() > 0.9:
                    uv[:, 1] = cand
                    break
        iv = np.clip(uv[:, 1].astype(int), 0, mask.shape[0] - 1)
        iu = np.clip(uv[:, 0].astype(int), 0, mask.shape[1] - 1)
        uv = uv[mask[iv, iu]]
        if len(uv) >= 3:
            controls.append(isw.Feature("line", uv, ln.confidence, dict(ln.info, control=args.mode)))
    records, _ = isw.features_to_3d(controls, xyz, valid, (w2g, w2g), args.coord_scale)
    records = isw.split_by_turns(records, v.umbilicus_center(args.umbilicus), args.max_turns)
    seg = os.path.basename(os.path.normpath(args.segment_dir))
    cols, stats = isw.build_collections(records, seg, {"control": args.mode}, 5)
    for c in cols.values():
        c["metadata"]["evidence"] = f"control_{args.mode}"
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"vc_pointcollections_json_version": "1", "collections": cols}, f)
    print(f"{args.mode}: lines {len(lines)} -> {stats}")


if __name__ == "__main__":
    main()
