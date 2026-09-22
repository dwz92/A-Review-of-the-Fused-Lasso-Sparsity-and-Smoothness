# A Review of the Fused Lasso Sparsity and Smoothness

This repo contains a review of Tibshirani, Saunders, Rosset, Zhu & Knight (2005), *"Sparsity and Smoothness via the Fused Lasso" (JRSS-B, Section 8)*, together with a replication of the paper's Section 8 simulation study (lasso vs. fused lasso on protein-mass-spectroscopy-like data), run for 100 Monte Carlo replications.

**Authors:** Qi Er (Emma) Teng, Yiding Shan

## Contents

```
.
├── README.md
├── A_Review_of_the_Fused_Lasso_Sparsity_and_Smoothness.pdf    # our write-up (review + replication)
├── replication_code/
│   ├── Replication_simulation_GEN.py                       # simulation code (see below)
│   ├── figures
│   │   ├── fig3_coefficient_plot.png                            # generated: coefficient-profile figure
│   │   ├── Table3Rep.txt                                        # generated: replicated Table 3
├── JNCI_Data_7-3-02/                                    # raw patient spectra (see "Data")
│   ├── Benign PSA Greater 4/
│   ├── No Evidence of Disease PSA Less 1/
│   ├── Prostate Cancer PSA 4-10/
│   └── Prostate Cancer PSA Greater 10/
└── Tibshirani-JRSSB05.pdf   # original paper (Tibshirani et al., 2005)
```

## Data

We use the publicly available prostate cancer serum proteomic data set from Petricoin/Paweletz et al. (2002), *"Serum proteomic patterns for detection of prostate cancer"* (https://home.ccr.cancer.gov/ncifdaproteomics/pdf/JNCI_prostate_paper.pdf), rather than the original Adam et al. (2003) data set used in the paper, which we could not obtain.

Patient spectra are organized into four diagnostic groups (see `GROUPS` in the script), used only to build a realistic feature-correlation structure — the simulated response is entirely synthetic and independent of the disease labels.

Download the data set and place the four group folders under `JNCI_Data_7-3-02/` (or update `DATA_DIR` at the top of `Replication_simulation_GEN.py` to point at wherever you keep it). Each patient file is a two-column CSV of `mz, intensity`.

## Requirements

```
python >= 3.9
numpy
pandas
matplotlib
scikit-learn
cvxpy            # with the CLARABEL solver
```

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn cvxpy
```

## Running the simulation

```bash
python Replication_simulation_GEN.py
```

This will:
1. Load and preprocess the patient spectra (filter to m/z ≥ 2000, block-average in groups of 20).
2. Subsample 100 patients and the first 518 usable features, standardize them.
3. Run 100 Monte Carlo replications, each generating a synthetic piecewise-constant `β`, a response `y = Xβ + ε`, fitting the lasso and the fused lasso, and recording test error, sensitivity, and specificity.
4. Write the summary table to `Table3Rep.txt` and save a representative coefficient-profile figure to `fig3_coefficient_plot.png`.

All randomness is controlled by `RAND_SEED = 315` at the top of the script, so re-running it reproduces the numbers below exactly.

## What the simulation does

Following Section 8 of the paper:

- **Feature matrix:** block-averaged, standardized spectral features used purely to give the design matrix a realistic correlation structure.
- **True coefficients β:** 1–10 non-overlapping blocks, block length ~ Uniform{1,...,100}, block value ~ N(0,1) (clipped to [-2, 2]), placed at random positions; remaining coefficients are 0.
- **Response:** `y = Xβ + ε`, `ε ~ N(0, 2.5²)`.
- **Lasso:** `sklearn.linear_model.Lasso`, selected over a 40-point log-spaced grid of `α` by minimum test error (warm-started).
- **Fused lasso:** solved in penalized form with `cvxpy` (CLARABEL solver), `min (1/2n)||y - Xβ||² + λ1||β||1 + λ2||Dβ||1`, with `λ1` fixed at the lasso's selected value
  and `λ2` searched over a 12-point log-spaced grid around `λ1`.
- **Fused lasso (true s1, s2):** fused lasso refit using λ's back-calculated to match the true `β`'s L1-norm and total-variation.
- 100 Monte Carlo runs, 50/50 train/test split each run.

### Adaptations from the original paper

The publicly available data set has only **518** usable features after the paper's preprocessing pipeline, versus the paper's ~1000, and a different class balance (which does not affect the results, since the response is synthetic). Two adjustments were needed to keep the simulation stable and comparable:

- **Nonzero fraction cap:** the total fraction of nonzero true coefficients is capped at 10% of `p`, so specificity stays well-defined with the smaller feature set.
- **Signal rescaling:** because block-averaged features are highly correlated, `Xβ`'s scale varied wildly across runs; `β` is rescaled after generation so `std(Xβ) = 2.5`, keeping the signal-to-noise ratio at ≈1 (matching the paper's ~50%-variance-explained target) in every run.
- The paper's LAR-based path search and SQOPT solver are replaced with a grid search over `α` (lasso) and `λ2` (fused lasso) using `scikit-learn` and `cvxpy`/CLARABEL, since SQOPT is not publicly available.
- 100 Monte Carlo runs are used instead of the paper's 20, for tighter standard errors.

## Results

**Our replication (100 runs):**

| Method                    | Test Error       | Sensitivity   | Specificity   |
|---------------------------|------------------|---------------|---------------|
| Lasso                     | 411.993 (11.492) | 0.070 (0.004) | 0.980 (0.001) |
| Fused lasso               | 387.253 (9.974)  | 0.400 (0.031) | 0.939 (0.008) |
| Fused lasso (true s1, s2) | 446.935 (13.138) | 0.111 (0.010) | 0.978 (0.002) |

**Paper's Table 3 (20 runs), for reference:**

| Method                    | Test Error      | Sensitivity   | Specificity   |
|---------------------------|-----------------|---------------|---------------|
| Lasso                     | 265.194 (7.957) | 0.055 (0.009) | 0.985 (0.003) |
| Fused lasso               | 256.117 (7.450) | 0.478 (0.082) | 0.693 (0.072) |
| Fused lasso (true s1, s2) | 261.380 (8.724) | 0.446 (0.045) | 0.832 (0.018) |

The central qualitative finding replicates: the fused lasso recovers far more of the true nonzero coefficients than the lasso (higher sensitivity) at a modest specificity cost, matching the paper's conclusion that ordering-aware regularization better captures block-structured signals. Test errors run higher in our replication mainly because of the smaller feature set (518 vs. ~1000); see the report for full discussion.

`fig3_coefficient_plot.png` shows lasso vs. fused lasso coefficient estimates for a representative run, with the true nonzero m/z block shaded in yellow.

## Reference

Tibshirani, R., Saunders, M., Rosset, S., Zhu, J. and Knight, K. (2005). Sparsity and smoothness via the fused lasso. *Journal of the Royal Statistical Society: Series B*, 67(1), 91–108.