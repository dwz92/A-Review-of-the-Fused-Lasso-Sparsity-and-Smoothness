"""
Replication of Section 8 (Simulation Study) from:
    "Sparsity and smoothness via the fused lasso" Tibshirani et al., JRSS-B 2005

Paper's setup:

    - Feature matrix X: first 1000 features from the prostate cancer data,
    using a random subset of 100 patients.

    - Coefficients beta: randomly generated as piecewise constant blocks.
        - Number of blocks: Uniform{1,...,10}
        - Block lengths: Uniform{1,...,100}
        - Block values: N(0,1)
        All placed at non-overlapping random positions.

    - Response: y = X @ beta + epsilon, epsilon = 2.5 * N(0,1)

Paper's Table 3 reference values:

    Lasso:                    265.2 (7.96)  sens=0.055 (0.009)  spec=0.985 (0.003)
    Fused lasso:              256.1 (7.45)  sens=0.478 (0.082)  spec=0.693 (0.072)
    Fused lasso (true s1,s2): 261.4 (8.72)  sens=0.446 (0.045)  spec=0.832 (0.018)

Our adaption:
    - Feature natrux X: Reduced to 518 because the paper had more dataset than us, after data processing and blocking, we only have 518 available feature for use.
    - Beta:
    - Response: 
"""

### Basic Set Up ###
import os
import time
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import cvxpy as cp


DATA_DIR = r"JNCI_Data_7-3-02/"
BLOCK = 20
N_SIM = 100 # Monte Carlo runs (paper: 20)
N_PATIENTS = 100 # patients to subsample (paper: 100)
P_FEATURES = 518 # We only have 518 in the dataset we found (paper: 1000)
NOISE_SD = 2.5
NONZERO_TOL = 1e-4 # threshold to call a coefficient non-zero
TRAIN_FRAC = 0.50
L1_GRID_N = 40
L2_GRID_N = 12
RAND_SEED = 315
MAX_NONZERO_FRAC = 0.10 # Cap nonzero fraction so specificity is not Nan and setting stays sparse (paper had p=1000, we have 518)

GROUPS = {
    "Benign PSA Greater 4":0,
    "No Evidence of Disease PSA Less 1":0,
    "Prostate Cancer PSA 4-10":1,
    "Prostate Cancer PSA Greater 10":1,
}



### Helpers ###
def data_overview(patients):
    print(f"\nTotal patients loaded: {len(patients)}")
    print(f"\nTotal m/z points: {len(patients[0][0])}")
    print(f"Class distribution:")
    labels_all = [s[2] for s in patients]
    print(f"Healthy/Benign (0): {labels_all.count(0)}")
    print(f"Cancer         (1): {labels_all.count(1)}")


def summarise(vals):
    a = np.array(vals)
    return a.mean(), a.std() / np.sqrt(len(a))


def TABLE_3():
    print("=" * 72)
    print("\nPaper's Table 3 (reference, 20 runs):")
    print("  Lasso                          265.194 (7.957)   0.055 (0.009)   0.985 (0.003)")
    print("  Fused lasso                    256.117 (7.450)   0.478 (0.082)   0.693 (0.072)")
    print("  Fused lasso (true s1,s2)       261.380 (8.724)   0.446 (0.045)   0.832 (0.018)")


def model_summary(results, saveto="Table3Rep.txt"):
    n_completed = len(results["lasso"]["err"])

    TABLE3 = [("lasso","Lasso"),
              ("fused","Fused lasso"),
              ("fused_true", "Fused lasso (true s1,s2)")]

    original_stdout = sys.stdout 

    with open(saveto, "w+") as f:
        sys.stdout = f
        print("\n" + "=" * 72)
        print(f"TABLE 3 REPLICATION: Result of simulation study over {n_completed} runs.")
        print(f"(Standard errors are given in parentheses)")
        print("=" * 72)

        print(f"{'Method':<32} {'Test Error':>20} {'Sensitivity':>14} {'Specificity':>14}")
        print("-" * 72)

        for key, name in TABLE3:
            e_m, e_se = summarise(results[key]["err"])
            sn_m, sn_se = summarise(results[key]["sens"])
            sp_m, sp_se = summarise(results[key]["spec"])
            print(f" {name:<30} {e_m:8.3f} ({e_se:.3f}) {sn_m:.3f} ({sn_se:.3f})   {sp_m:.3f} ({sp_se:.3f})")

        print("=" * 72)

        TABLE_3()

        sys.stdout = original_stdout

        


def fig_3(rep_run):
    beta_true = rep_run["beta_true"]
    lasso_coef = rep_run["lasso_coef"]
    fl_coef = rep_run["fl_coef"]
    sim_num = rep_run["sim_num"]
    n_true_nz = int((np.abs(beta_true) > NONZERO_TOL).sum())
 
    x = mz_axis # real m/z values on x-axis from og data
 
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(
        f"Figure 3 Replication: Coefficient profiles vs m/z "
        f"(simulation run #{sim_num}, p = {P_FEATURES}, "
        f"true nonzero = {n_true_nz})",
        fontsize=13, fontweight="bold",
    )
 
    panels = [
        (axes[0], lasso_coef, "Lasso",
         rep_run["l_err"], rep_run["l_sens"],  rep_run["l_spec"]),
        (axes[1], fl_coef, "Fused lasso",
         rep_run["fl_err"], rep_run["fl_sens"], rep_run["fl_spec"]),
    ]
 
    for ax, coef, label, err, sens, spec in panels:
        ax.axhline(0, color="grey", lw=0.6, ls="--", zorder=1)
 
        # True coefficient profile: solid black line + dots
        ax.plot(x, beta_true, color="black", lw=1.2, zorder=3, label="True β")
        ax.scatter(x, beta_true, color="black", s=5, zorder=4)
 
        # Estimated coefficients: red dots
        # Brighter for nonzero estimates, faint for zeros
        nz = np.abs(coef) > NONZERO_TOL
        ax.scatter(x[nz], coef[nz], color="tab:red", s=9, alpha=0.85, zorder=5,
                   label=f"{label} estimate  (nonzero n={nz.sum()})")
        ax.scatter(x[~nz], coef[~nz], color="tab:red", s=4, alpha=0.20, zorder=4)
 
        ax.set_ylabel("Coefficient value", fontsize=10)
        ax.set_title(
            f"{label}   |   test error = {err:.1f}   sensitivity = {sens:.3f}   specificity = {spec:.3f}",
            fontsize=10,
        )
        ax.legend(loc="upper right", fontsize=9, markerscale=1.8)
        ax.spines[["top", "right"]].set_visible(False)
 
    # Shade true nonzero m/z regions in yellow across both panels
    nz_mask = np.abs(beta_true) > NONZERO_TOL
    in_block = False
    b_start = None
    for i, nz in enumerate(nz_mask):
        if nz and not in_block:
            b_start = x[i]
            in_block = True
        elif not nz and in_block:
            for ax in axes:
                ax.axvspan(b_start, x[i - 1], color="gold", alpha=0.22, zorder=0)
            in_block = False
    if in_block:
        for ax in axes:
            ax.axvspan(b_start, x[-1], color="gold", alpha=0.22, zorder=0)
 
    axes[1].set_xlabel("m/z", fontsize=11)
 
    plt.tight_layout()
    fig_path = "fig3_coefficient_plot.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {fig_path}")
    plt.show()


### Load and Preprocess Dataset ###
def patient_group(path, label):
    group_patients = []

    for f in os.listdir(path):
        file = os.path.join(path, f)
        if os.path.isfile(file):

            # print(f"Loading patient files: {file}")

            try:
                df = pd.read_csv(file, header=0)
                df.columns = ['mz', 'intensity']
                group_patients.append((df['mz'].values,
                                df['intensity'].values,
                                label,
                                path))
            except Exception as e:
                print(f"File: {file} encounterred error: {e}")

    return group_patients


def load_data(dirpath, groups):
    res = [] # N x 3 array of MZ array, intensity array, label

    for group in groups:
        group_path = os.path.join(dirpath, group)
        res.extend(patient_group(group_path, GROUPS[group]))
    
    return res


def preprocessing(data, block=BLOCK):

    mz, intensity, label, path = data

    mask = mz >= 2000
    mz_filtered = mz[mask]
    sig_filtered = intensity[mask]

    # Trim to multiple of block size, then block-average
    n = (len(mz_filtered) // block) * block
    mz_blocked = mz_filtered[:n].reshape(-1, block).mean(axis=1)
    sig_blocked = sig_filtered[:n].reshape(-1, block).mean(axis=1)

    return mz_blocked, sig_blocked


### Model Coefficients and Fitting ###
def generate_beta(p, rng, max_nonzero_frac=MAX_NONZERO_FRAC):
    """
    """
    beta = np.zeros(p)
    n_blocks = rng.integers(1, 11)
    positions = list(range(p))
    max_nonzero = int(max_nonzero_frac * p)
    n_nonzero = 0

    for _ in range(n_blocks):
        if len(positions) == 0:
            break
        
        budget = max_nonzero - n_nonzero
        if budget <= 0:
            break

        block_len = min(rng.integers(1, 101), len(positions))  # Uniform{1,...,100}

        max_start = len(positions) - block_len
        if max_start < 0:
            break

        start_idx = rng.integers(0, max_start + 1)
        chosen = positions[start_idx : start_idx + block_len]
        value = rng.standard_normal()
        value = np.clip(value, -2, 2)

        for idx in chosen:
            beta[idx] = value
        
        n_nonzero += len(chosen)
        chosen_set = set(chosen)
        positions = [pp for pp in positions if pp not in chosen_set]
 
    return beta


def compute_metrics(beta_hat, beta_true, y_test, X_test):
    """
    """
    resid = y_test - X_test @ beta_hat
    test_err = float(np.dot(resid, resid))
 
    true_nz = np.abs(beta_true) > NONZERO_TOL
    hat_nz = np.abs(beta_hat)  > NONZERO_TOL
 
    # sensitivity = TP / (TP + FN)
    sensitivity = float((true_nz & hat_nz).sum() / true_nz.sum()) if true_nz.sum() > 0 else np.nan
 
    # specificity = TN / (TN + FP)
    true_z = ~true_nz
    specificity = float((true_z & ~hat_nz).sum() / true_z.sum()) if true_z.sum() > 0 else np.nan
 
    return test_err, sensitivity, specificity


def fit_lasso(X_tr, y_tr, X_te, y_te, n_alphas=L1_GRID_N):
    """
    """
    alpha_max = np.max(np.abs(X_tr.T @ y_tr)) / len(y_tr)
    alphas = np.logspace(np.log10(alpha_max * 0.05), np.log10(alpha_max), n_alphas)[::-1] # the prev was too small, picking up noises
 
    best_err = np.inf
    best_coef = None
    best_alpha = None
    
    model = Lasso(alpha=alphas[0], max_iter=50000, fit_intercept=False, tol=1e-5, warm_start=True)
 
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    for alpha in alphas:
        model.set_params(alpha=alpha)
        model.fit(X_tr, y_tr)

        err = float(np.sum((y_te - X_te @ model.coef_) ** 2))

        if err < best_err:
            best_err, best_coef, best_alpha = err, model.coef_.copy(), alpha
 
    return best_coef, best_alpha


def fit_fused_lasso(X_tr, y_tr, l1, l2):
    """
    """
    p = X_tr.shape[1]
    beta = cp.Variable(p)
    D = np.diff(np.eye(p), axis=0) # (p-1) × p first difference matrix
 
    n = X_tr.shape[0]
    obj = cp.Minimize(
        (1/(2*n)) * cp.sum_squares(X_tr @ beta - y_tr)
        + l1 * cp.norm1(beta)
        + l2 * cp.norm1(D @ beta)
    )
    
    prob = cp.Problem(obj)

    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        return None
    return beta.value if beta.value is not None else None


def fit_fused_lasso_grid(X_tr, y_tr, X_te, y_te, l1_fixed, n_l2=L2_GRID_N):
    """
    """
    l2_grid = l1_fixed * np.logspace(-2, 2, n_l2)
    best_err = np.inf
    best_coef = None

    for l2 in l2_grid:
        coef = fit_fused_lasso(X_tr, y_tr, l1_fixed, l2)

        if coef is None:
            continue

        err = float(np.sum((y_te - X_te @ coef) ** 2))

        if err < best_err:
            best_err = err
            best_coef = coef.copy()

    return best_coef



### Load data ###
print("Loading prostate cancer data...")

patients = load_data(DATA_DIR, GROUPS)
processed = [preprocessing(p) for p in patients]
min_len = min(len(p[1]) for p in processed)
X_full = np.array([p[1][:min_len] for p in processed]) # (all_patients, all_features)

# Save the m/z axis from the first patient for plotting (same grid for all)
mz_axis = processed[0][0][:min_len] # shape (P_FEATURES,)

print(f"Full matrix: {X_full.shape} => keeping first {P_FEATURES} features from {N_PATIENTS} randomly sampled patients")

# Paper uses first 1000 features and a random subset of 100 patients, we only have 518 feature here
rng_data = np.random.default_rng(RAND_SEED)
patient_idx = rng_data.choice(X_full.shape[0], size=N_PATIENTS, replace=False)
X_base = X_full[patient_idx, :P_FEATURES]

# Standardize base features, the paper used predictors standardized to mean 0, unit variance
X_base_scaler = StandardScaler()
X_base = X_base_scaler.fit_transform(X_base)

print(f"Simulation base matrix X_base: {X_base.shape}")


### Monte Carlo ###
results = {
    "lasso": {"err": [], "sens": [], "spec": []},
    "fused": {"err": [], "sens": [], "spec": []},
    "fused_true": {"err": [], "sens": [], "spec": []},
}

rep_run = None

print(f"\nRunning {N_SIM} Monte Carlo simulations (p={P_FEATURES}, N={N_PATIENTS})...")
t_start = time.time()

for sim in range(N_SIM):
    rng = np.random.default_rng(RAND_SEED + sim + 1)

    # Generate Beta
    beta_true = generate_beta(P_FEATURES, rng)

    # Get response (also scale the noise according to paper)
    signal = X_base @ beta_true
    signal_std = np.std(signal)
    if signal_std > 0:
        beta_true = beta_true * (NOISE_SD / signal_std)

    y_all = X_base @ beta_true + NOISE_SD * rng.standard_normal(N_PATIENTS)

    # Model Train
    n_train = int(N_PATIENTS * TRAIN_FRAC)
    idx = rng.permutation(N_PATIENTS)
    tr_idx, te_idx = idx[:n_train], idx[n_train:]

    X_tr, X_te = X_base[tr_idx], X_base[te_idx]
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]

    # Lasso
    lasso_coef, best_l1 = fit_lasso(X_tr, y_tr, X_te, y_te)

    if lasso_coef is None:
        print(f"Run {sim+1:3d}: lasso failed, skipping")
        continue

    l_err, l_sens, l_spec = compute_metrics(lasso_coef, beta_true, y_te, X_te)
    results["lasso"]["err"].append(l_err)
    results["lasso"]["sens"].append(l_sens)
    results["lasso"]["spec"].append(l_spec)


    # Fused Lasso
    fl_coef = fit_fused_lasso_grid(X_tr, y_tr, X_te, y_te, l1_fixed=best_l1)

    if fl_coef is None:
        fl_coef = lasso_coef

    fl_err, fl_sens, fl_spec = compute_metrics(fl_coef, beta_true, y_te, X_te)
    results["fused"]["err"].append(fl_err)
    results["fused"]["sens"].append(fl_sens)
    results["fused"]["spec"].append(fl_spec)


    # Fused lasso at TRUE
    s1_true = float(np.sum(np.abs(beta_true)))
    s2_true = float(np.sum(np.abs(np.diff(beta_true))))
    s1_lasso = float(np.sum(np.abs(lasso_coef)))

    true_l1 = best_l1 * s1_true / max(s1_lasso, 1e-8)
    true_l2 = true_l1 * s2_true / max(s1_true, 1e-8)

    ft_coef = fit_fused_lasso(X_tr, y_tr, true_l1, true_l2)

    if ft_coef is None:
        ft_coef = fl_coef
    
    ft_err, ft_sens, ft_spec = compute_metrics(ft_coef, beta_true, y_te, X_te)
    results["fused_true"]["err"].append(ft_err)
    results["fused_true"]["sens"].append(ft_sens)
    results["fused_true"]["spec"].append(ft_spec)

    # Train update
    elapsed = time.time() - t_start

    print(f"Run {sim+1:3d}/{N_SIM}  |  "
          f"lasso err={l_err:7.1f} sens={l_sens:.3f} spec={l_spec:.3f}  |  "
          f"fused err={fl_err:7.1f} sens={fl_sens:.3f} spec={fl_spec:.3f}  |  "
          f"elapsed {elapsed:.0f}s")
    
    # Save the first decent coefficients
    if rep_run is None:
        nz_count = int((np.abs(beta_true) > NONZERO_TOL).sum())
        z_count = P_FEATURES - nz_count

        if nz_count >= 5 and z_count >= 5 and not np.isnan(l_spec) and not np.isnan(fl_spec):
            rep_run = {
                'beta_true': beta_true.copy(),
                'lasso_coef': lasso_coef.copy(),
                'fl_coef': fl_coef.copy(),
                'l_err': l_err, 'l_sens': l_sens, 'l_spec': l_spec,
                'fl_err': fl_err, 'fl_sens': fl_sens, 'fl_spec': fl_spec,
                'sim_num': sim + 1,
            }



### Model Summary ###
model_summary(results)
TABLE_3()

# fig 3
if rep_run is None:
    print("\nWarning: all runs had NaN.")
else:
    fig_3(rep_run)