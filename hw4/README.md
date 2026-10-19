# HW4 — colour K-means

The original comb-sampled Lloyd iterations now handle exact multiples of K and empty clusters safely, accept a reproducible seed, and recompute assignments after the last centroid update. Colour block sampling and full-size reconstruction live in `colorquant.cpp`.

```bash
build/bin/colors project1/cathedral.jpeg 8 --seed 1 --out results/kmeans --headless
```

Run from the root after [building](../README.md#run-everything). `--block`, `--max-iter`, and `--seed` control the quality/computation trade-off. The output includes a quantised image and a palette strip. This is seeded comb sampling, not K-means++.

![Eight-colour result](../docs/gallery/kmeans.jpg)

Tests cover the division-by-zero regression, K bounds, empty clusters, deterministic results, final assignments and non-divisible image dimensions.
