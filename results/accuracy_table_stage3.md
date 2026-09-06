# Stage 3 — accuracy (top-1 %, complete official test split)

Generated from `results/runs.csv`. Bracketed values are the change against the
Stage 1 **linear probe** for the same dataset, encoder and K.

## aircraft · dinov2_vits14

**K = 10**

| strategy | variant | top-1 | delta |
|---|---|---|---:|
| — | Stage 1 linear probe | 50.56 | |
| end-to-end | fm_e2e_lam0 | 50.58 ± 0.77 | +0.02 |
| end-to-end | fm_e2e_lam0.01 | 50.58 ± 0.02 | +0.02 |
| end-to-end | fm_e2e_lam0.1 | 50.58 ± 0.09 | +0.02 |
| classifier-guided | fm_guided_eta0.1 | 50.55 ± 0.15 | -0.01 |
| classifier-guided | fm_guided_eta1 | 50.53 ± 0.08 | -0.03 |
| classifier-guided | fm_guided_eta10 | 50.53 ± 0.09 | -0.02 |

**K = full**

| strategy | variant | top-1 | delta |
|---|---|---|---:|
| — | Stage 1 linear probe | 67.69 | |
| end-to-end | fm_e2e_lam0 | 67.84 ± 0.19 | +0.15 |
| end-to-end | fm_e2e_lam0.1 | 67.68 ± 0.09 | -0.01 |


## dtd · resnet18

**K = 10**

| strategy | variant | top-1 | delta |
|---|---|---|---:|
| — | Stage 1 linear probe | 52.09 | |
| end-to-end | fm_e2e_lam0 | 51.95 ± 0.39 | -0.14 |
| end-to-end | fm_e2e_lam0.01 | 52.29 ± 0.45 | +0.20 |
| end-to-end | fm_e2e_lam0.1 | 52.55 ± 0.58 | +0.46 |
| classifier-guided | fm_guided_eta0.1 | 52.09 ± 0.41 | -0.00 |
| classifier-guided | fm_guided_eta1 | 52.16 ± 0.35 | +0.07 |
| classifier-guided | fm_guided_eta10 | 52.30 ± 0.48 | +0.21 |

**K = full**

| strategy | variant | top-1 | delta |
|---|---|---|---:|
| — | Stage 1 linear probe | 62.55 | |
| end-to-end | fm_e2e_lam0 | 62.15 ± 0.71 | -0.41 |
| end-to-end | fm_e2e_lam0.1 | 63.71 ± 0.63 | +1.15 |

