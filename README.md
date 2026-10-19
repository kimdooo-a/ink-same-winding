# ink-same-winding

Same-winding constraints for the [spiral fit](https://github.com/ScrollPrize/villa/tree/main/spiral-fitting),
generated from **published ink predictions** on flattened segments: text lines and
intercolumnar gaps are traced on the 2D ink image, mapped through the segment's
tifxyz grid to 3D, and written as a `same_windings.json` point-collection file that
`spiral-fitting` loads as is. A validator measures every output against the
published verified patches and annotations, with ±1-winding negative controls.

> **Ink as same-winding evidence** — *Easily identifiable correct letters, rows, or columns are a natural
> source of same-winding constraints (and are likely the most easily verified).*
>
> **Intercolumnar gaps** — *The blank vertical spaces between columns of text can form long, recognizable
> paths across the papyrus surface. Like ink, they could provide same-winding evidence that is easy to
> identify and verify.*
>
> — [scrollprize.org/open_problems/winding_annotations](https://scrollprize.org/open_problems/winding_annotations), "Things we have not tried"

This tool only re-uses ink predictions that are already public. It does not read, transcribe or
publish text; outputs contain coordinates and confidences only.

## What it does

```
ink prediction (2D, aligned to the segment grid)          tifxyz x/y/z.tif (level-0 voxels)
        │ column profile → intercolumnar gaps                       │
        │ row profiles in 12 mm strips → peaks → linked lines       │
        └──────── (u, v) samples, 1 mm apart, only where letters are ┘
                         │ bilinear lookup on the grid, × 0.25 (→ 9.6 µm level-2 frame)
                         │ cut into ≤ 0.25-turn chunks around the umbilicus (optional)
                         │ drop isolated hits on verified patches (optional)
                         ▼
              same_windings.json  (vc_pointcollections_json_version 1, no wind_a → same-winding)
```

Each collection carries `metadata.evidence` (`ink_text_line` | `intercolumnar_gap` | `edge_margin`),
`segment_id`, `confidence`, measured line spacing / gap width, a VC3D-style `hv_classification`
(`H` for lines, `V` for gaps), and the grid (u, v) of its first and last point. Point ids follow the
feature (left→right, top→bottom) because spiral-fitting reads unattached collections as ordered strips.
Chains are cut wherever no letter is seen for 4 mm, so neighbouring points stay well under the
`|Δθ| < π` the strip loss assumes.

## Install

```sh
python -m venv .venv && . .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt                   # numpy scipy tifffile imagecodecs pillow
# only for smoke_load.py (imports spiral-fitting's own loader):
pip install -r requirements-smoke.txt
```

CPU only. One banner segment (~285 MB of downloads) runs in about 10 s; validation against
verified patches takes 1–3 min once the patches are cached.

## Usage

```sh
# generate (segment dir laid out like the open-data bucket: ink-detection/…, mesh/…-on-<volume>-*.tifxyz/)
python ink_same_winding.py data/PHercParis4/segments/20260623144957-w053-058 \
    -o out/w053-058.same_windings.json --debug-png out/w053-058.png \
    --umbilicus data/spiral-input/umbilicus.json --max-turns 0.25

# optional: also drop points that touch a verified patch with < 3 points of the same collection
#   --min-patch-pts 3 --patch-meta-dir … --patch-root … --patch-cache …

# check that spiral-fitting loads it (needs a checkout of ScrollPrize/villa)
python smoke_load.py <villa>/spiral-fitting out/w053-058.same_windings.json

# validate against verified patches (+ negative control: --radial-shift 19 ≈ one winding)
python validate_against_patches.py out/w053-058.same_windings.json \
    --patch-meta-dir data/patch_meta --patch-root https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/verified_patches_100226 \
    --patch-cache data/patches --umbilicus data/spiral-input/umbilicus.json \
    --relative data/spiral-input/relative_windings.json --same data/spiral-input/same_windings.json \
    --tol 2.5 5 --min-patch-points 3
```

`examples/run_example.sh` downloads one banner and runs the generator end to end.

| tool | purpose |
|---|---|
| `ink_same_winding.py` | the generator |
| `validate_against_patches.py` | attach rate, contradictions vs. published annotations, self-consistency, angle checks |
| `make_controls.py` | same-budget control chains (between text lines, or at random rows) |
| `ink_value.py` | segment-vs-patch winding consistency along the segment, and patch-graph coverage |
| `diagnose_contradictions.py` | lists each patch pair a collection disagrees on, with where it happens |
| `smoke_load.py` | loads an output with `spiral-fitting`'s `load_point_collection` |

## Coordinates (measured, not assumed)

* Point order is `p = [x, y, z]`: in the public `same_windings.json` only `p[2]` exceeds 8,173
  (max 14,068), the level-2 x/y extent of volume `20260411134726` (level-0 shape `[75784, 32693, 32693]`
  z,y,x); spiral-fitting reverses `p` to zyx (`point_collection.py`, `fit_spiral.py`).
* Units are 9.6 µm level-2 voxels. The `…-on-20260411134726-2.4um.tifxyz` grids are level-0 and are
  multiplied by 0.25. Points land on verified patches at a median 1.17 voxels (p10 0.12).
* The ink image is aligned to that transformed grid: 20.000 px per cell for the full TIF, 2.500 for the
  `-ds8.jpg`. The `intermediate/tifxyz_normalized` grid is *not* an exact ×4 of it (affine residual
  ≈ 8 level-0 voxels), so it is not used.
* Winding coordinate used by the validator: `W = unwrap(atan2(y − cy, x − cx)) / 2π` around
  `umbilicus.json`; on PHercParis4 the radius grows with W (+6.7 … +21.8 voxels per turn along the
  6-turn banner) and `wind_a` in `relative_windings.json` grows outward (corr. with radius 0.994).

## Results on PHercParis4 (five banner segments, 20,140 points)

Segments `20260623144224-w046-052`, `20260623144957-w053-058`, `20260623145652-w059-063`,
`20260623150417-w064-068` and a second trace of the same windings, `20260701183128-w053-058`.
Inputs: the `new_canon_autoresearch_recipe` ink prediction (`-ds8.jpg`). Reference: 22,563 verified
patches (`verified_patches_100226`) plus the published `relative_windings.json` / `same_windings.json`.
Tolerance 2.5 voxels is spiral-fitting's default `pcl_link_distance_tolerance`; a patch counts only if a
collection touches it with ≥ 3 points.

| condition | attach rate (≤ 2.5 vox) | patch pairs vs. published annotations: agree / contradict | same patch revisited at a different winding |
|---|---|---|---|
| **ink lines + gaps** | 36.8 % | 48 / 2 (4.0 %) | 13 / 892 (1.5 %) |
| control: between text lines | 35.9 % | 32 / 0 | 11 / 757 (1.5 %) |
| control: random rows | 29.2 % | 25 / 1 | 14 / 676 (2.1 %) |
| control: ink points moved +19 vox (≈ one winding out) | 17.7 % | 12 / 3 (20 %) | 77 / 363 (21.2 %) |
| control: moved −19 vox | 17.4 % | 3 / 3 (50 %) | 77 / 378 (20.4 %) |

Per segment (tolerance 2.5; attach rate · reference pairs agree/contradict · inconsistent revisits):

| segment | columns · gaps · lines | collections · points | ink | +1 winding | −1 winding | between lines | random rows |
|---|---|---|---|---|---|---|---|
| w046-052 | 5 · 4 · 62 | 168 · 2,659 | 44.8 % · 6/0 · 3/134 | 17.1 % · 1/0 · 11/47 | 21.3 % · 1/0 · 21/68 | 41.1 % · 4/0 · 3/114 | 31.4 % · 1/0 · 2/99 |
| w053-058 | 6 · 5 · 80 | 240 · 4,250 | 39.6 % · 5/1 · 2/189 | 20.6 % · 4/2 · 26/95 | 17.6 % · 0/1 · 22/83 | 38.8 % · 6/0 · 1/159 | 30.3 % · 1/1 · 5/135 |
| w059-063 | 7 · 5 · 67 | 187 · 3,660 | 55.4 % · 26/0 · 1/256 | 21.6 % · 2/0 · 14/68 | 21.0 % · 1/0 · 9/70 | 54.5 % · 16/0 · 0/203 | 52.6 % · 20/0 · 2/225 |
| w064-068 | 4 · 3 · 63 | 156 · 3,139 | 18.2 % · 1/0 · 1/69 | 11.5 % · 0/0 · 1/31 | 15.4 % · 1/1 · 2/48 | 17.5 % · 1/0 · 2/65 | 17.3 % · 1/0 · 1/61 |
| w053-058 (2nd trace) | 7 · 6 · 136 | 366 · 6,432 | 30.3 % · 10/1 · 6/244 | 16.7 % · 5/1 · 25/122 | 14.7 % · 0/1 · 23/109 | 30.8 % · 5/0 · 5/216 | 20.5 % · 2/0 · 4/156 |

**What this shows**

* The checks have teeth: moving the same points by about one winding multiplies the revisit
  inconsistency ~14× and the contradiction rate 5–12×.
* The generated constraints agree with the published verified patches and annotations at the level the
  segment itself does.
* **The ink does not make the geometry more accurate.** Chains placed between the text lines on the same
  segment score the same; random rows score slightly lower only because some fall on worse parts of the
  surface. The accuracy belongs to the segment. What the ink contributes is *where* the constraints are and
  that each one is a visible line of letters or a blank column band a person can check in seconds
  (`--debug-png`).
* Coverage (`ink_value.py`, patches touched with ≥ 3 points): summed over the five segments (the two
  w053-058 traces counted separately) the constraints relate 345 verified-patch pairs; 314 of them (91 %) are
  not directly related by any published annotation, and 217 join patches that no chain of published
  relations connects (e.g. w053-058: the 54 patches it touches fall into 32 separate reference components
  before, 2 after). This mostly comes from turning a segment into constraints, not from the ink as such:
  per 1,000 points, collections that reach ≥ 2 patches are 9.7 for ink lines + gaps, 8.4 for chains between
  the lines and 6.8 for random rows. The vertical gap collections add z-spanning links that line-shaped
  controls do not have.
* Sheet-skip detection from ink could **not** be evaluated: along three traces (w046-052, w053-058 twice)
  the segment's winding offset to the largest verified band (`0000_top_band`, 454 cm²) stays the same in every
  0.05-turn bin (0 of 121–141 bins shift; vertex agreement 99.1–99.5 %), so there is no skip to detect.
  The band may share lineage with the banners, so this is not an independent proof of their correctness.
  Several other large verified patches (`anon_*`, `lt_*`, `kb_*`) do drift by whole turns against both the
  segment and the band; `ink_value.py` lists them as candidates for inspection (this may also be an
  artefact of unwrapping angles over those patch grids; not checked in 3D).

## Limitations

* Line detection is classical and finds roughly 50–60 % of the text lines; where an intercolumnar gap is
  missed, a "line" can run across two columns (still on the same sheet, but not one text line).
* Tested only on PHercParis4 banners at 9.6 µm; parameters (letter size, column width) are in mm and
  may need adjusting for other scrolls.
* A spiral fit with these constraints has not been run (needs the lasagna inputs and a GPU;
  `spiral-scroll.json` is not in the public dataset folder).
* Reference relations exist for only 1–26 patch pairs per segment, so the contradiction rates rest on
  small numbers; the revisit check (hundreds per segment) is the stronger signal.

## Data

All inputs are from the Vesuvius Challenge open data (license: CC BY-NC 4.0, see
`https://dl.ash2txt.org/LICENSE.txt`):

* segments and ink predictions: `s3://vesuvius-challenge-open-data/PHercParis4/segments/<id>/`
  (`https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/…`)
* spiral-input: `https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/`
  (`umbilicus.json`, `same_windings.json`, `relative_windings.json`, `verified_patches_100226/`)

No data is redistributed in this repository.

## License

Code: MIT (see `LICENSE`). Generated constraint files derived from the open data inherit its CC BY-NC 4.0 terms.
