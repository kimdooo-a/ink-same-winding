"""Smoke test: load a generated file with spiral-fitting's own loader.

usage: python smoke_load.py <spiral-fitting dir> <same_windings.json> [reference.json]
"""
import sys

import numpy as np


def main():
    sys.stdout.reconfigure(errors="replace")
    repo, path = sys.argv[1], sys.argv[2]
    sys.path.insert(0, repo)
    from point_collection import load_point_collection, normalise_pcl_winding_annotations
    for p in [path] + sys.argv[3:]:
        pcls = load_point_collection(p)
        assert pcls, f"loader returned nothing for {p}"
        normalise_pcl_winding_annotations(pcls)
        same = sum(1 for c in pcls.values() if not c['has_winding_annotations'])
        P = np.array([pt['p'] for c in pcls.values() for pt in c['points'].values()], float)
        wa = {float(pt['winding_annotation']) for c in pcls.values() for pt in c['points'].values()}
        print(f"{p}\n  collections={len(pcls)} same-winding={same} points={len(P)} "
              f"winding_annotation values={sorted(wa)}\n  p min={P.min(0).round(1)} max={P.max(0).round(1)}")


if __name__ == "__main__":
    main()
