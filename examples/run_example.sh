#!/usr/bin/env bash
# End-to-end example on one PHercParis4 banner segment (~285 MB download).
# Everything goes under $DATA (default ./data); outputs under $OUT (default ./out).
set -euo pipefail

DATA=${DATA:-./data}
OUT=${OUT:-./out}
PY=${PY:-python}
SEG_ID=${SEG_ID:-20260623144957-w053-058}
ID=${SEG_ID%%-*}

S3=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com
SPIRAL=https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4
SEG=PHercParis4/segments/$SEG_ID
MESH=mesh/$ID-on-20260411134726-2.4um.tifxyz
INK=PHercParis4-$ID-2.4um-0.22m-78keV-volume-20260411134726-20260417190342-new_canon_autoresearch_recipe-tile256-stride128-ds8.jpg

mkdir -p "$DATA/$SEG/$MESH" "$DATA/$SEG/ink-detection/downsampled" "$DATA/spiral-input" "$OUT"

# 1. segment: tifxyz grid (level-0 2.4 um voxels) + the published ink prediction (8x downsampled)
for f in meta.json x.tif y.tif z.tif; do
  [ -s "$DATA/$SEG/$MESH/$f" ] || curl -sfo "$DATA/$SEG/$MESH/$f" "$S3/$SEG/$MESH/$f"
done
[ -s "$DATA/$SEG/ink-detection/downsampled/$INK" ] || \
  curl -sfo "$DATA/$SEG/ink-detection/downsampled/$INK" "$S3/$SEG/ink-detection/downsampled/$INK"

# 2. spiral-input annotations (small)
for f in umbilicus.json same_windings.json relative_windings.json; do
  [ -s "$DATA/spiral-input/$f" ] || curl -sfLo "$DATA/spiral-input/$f" "$SPIRAL/$f"
done

# 3. generate same-winding constraints (spiral-input frame: level-2 9.6 um voxels, [x, y, z])
$PY ink_same_winding.py "$DATA/$SEG" -o "$OUT/$SEG_ID.same_windings.json" \
    --debug-png "$OUT/$SEG_ID.debug.png" \
    --umbilicus "$DATA/spiral-input/umbilicus.json" --max-turns 0.25

# 4. (optional) check against verified patches: needs the patch meta index first, e.g.
#    curl the directory listing of $SPIRAL/verified_patches_100226/ and fetch <name>/meta.json
#    into $DATA/patch_meta/<name>.json, then:
# $PY validate_against_patches.py "$OUT/$SEG_ID.same_windings.json" \
#     --patch-meta-dir "$DATA/patch_meta" --patch-root "$SPIRAL/verified_patches_100226" \
#     --patch-cache "$DATA/patches" --umbilicus "$DATA/spiral-input/umbilicus.json" \
#     --relative "$DATA/spiral-input/relative_windings.json" --same "$DATA/spiral-input/same_windings.json" \
#     --tol 2.5 5 --min-patch-points 3 --out "$OUT/$SEG_ID.validation.json"
