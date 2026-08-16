# Accuracy (top-1 %, complete official test split)

Generated from `results/runs.csv`. 18 rows x 3 settings = 54 cells, 0 missing.

| dataset | encoder | head | 5 | 10 | full |
|---|---|---|---|---|---|
| aircraft | dinov2_vits14 | fm_roll_T12 | 33.74 ± 0.89 | 43.44 ± 0.82 | 53.39 ± 0.58 |
| aircraft | dinov2_vits14 | fm_roll_T4 | 33.62 ± 0.68 | 43.31 ± 0.75 | 54.25 ± 0.59 |
| aircraft | dinov2_vits14 | fm_std_T12 | 33.18 ± 0.29 | 42.76 ± 0.72 | 57.63 ± 0.53 |
| aircraft | dinov2_vits14 | fm_std_T4 | 33.47 ± 0.41 | 43.21 ± 0.69 | 58.17 ± 0.86 |
| aircraft | dinov2_vits14 | linear | 36.66 ± 0.42 | 50.56 ± 0.12 | 67.69 ± 0.08 |
| aircraft | dinov2_vits14 | prototypes | 24.33 ± 0.24 | 27.51 ± 1.22 | 34.26 |
| aircraft | resnet18 | fm_roll_T12 | 14.61 ± 0.58 | 20.30 ± 0.97 | 25.75 ± 1.57 |
| aircraft | resnet18 | fm_roll_T4 | 14.89 ± 0.82 | 20.26 ± 0.88 | 25.50 ± 1.74 |
| aircraft | resnet18 | fm_std_T12 | 17.74 ± 0.30 | 23.69 ± 0.30 | 32.59 ± 0.40 |
| aircraft | resnet18 | fm_std_T4 | 17.36 ± 0.23 | 23.60 ± 0.37 | 32.33 ± 0.43 |
| aircraft | resnet18 | linear | 20.41 ± 0.35 | 26.90 ± 0.23 | 38.07 ± 0.27 |
| aircraft | resnet18 | prototypes | 16.39 ± 0.26 | 19.69 ± 0.20 | 25.50 |
| dtd | resnet18 | fm_roll_T12 | 40.53 ± 1.43 | 41.33 ± 0.88 | 55.80 ± 0.32 |
| dtd | resnet18 | fm_roll_T4 | 40.39 ± 1.35 | 40.89 ± 0.78 | 56.01 ± 0.27 |
| dtd | resnet18 | fm_std_T12 | 44.63 ± 0.97 | 48.33 ± 0.52 | 59.63 ± 0.05 |
| dtd | resnet18 | fm_std_T4 | 44.15 ± 0.84 | 47.68 ± 0.45 | 59.49 ± 0.08 |
| dtd | resnet18 | linear | 46.15 ± 0.71 | 52.09 ± 0.45 | 62.55 ± 0.51 |
| dtd | resnet18 | prototypes | 47.13 ± 0.61 | 51.56 ± 0.92 | 58.78 |
