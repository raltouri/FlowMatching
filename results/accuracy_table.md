# Stage 1 — accuracy (top-1 %, complete official test split)

Generated from `results/runs.csv`. 6 rows x 3 settings = 18 cells, 0 missing.

| dataset | encoder | head | 5 | 10 | full |
|---|---|---|---|---|---|
| aircraft | dinov2_vits14 | linear | 36.66 ± 0.42 | 50.56 ± 0.12 | 67.69 ± 0.08 |
| aircraft | dinov2_vits14 | prototypes | 24.33 ± 0.24 | 27.51 ± 1.22 | 34.26 |
| aircraft | resnet18 | linear | 20.41 ± 0.35 | 26.90 ± 0.23 | 38.07 ± 0.27 |
| aircraft | resnet18 | prototypes | 16.39 ± 0.26 | 19.69 ± 0.20 | 25.50 |
| dtd | resnet18 | linear | 46.15 ± 0.71 | 52.09 ± 0.45 | 62.55 ± 0.51 |
| dtd | resnet18 | prototypes | 47.13 ± 0.61 | 51.56 ± 0.92 | 58.78 |
