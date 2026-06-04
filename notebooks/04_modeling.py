"""
Buchwald-Hartwig reaction-yield prediction: modelling and honest evaluation.

Reproduces five results:
  1. Baseline random forest on Morgan fingerprints, random split.
  2. Same model on a harder split where whole additives are held out.
  3. Feature controls: fingerprints vs one-hot identities vs random features.
  4. Conformal 90% prediction intervals, and how coverage changes with the split.
  5. Learning curve: accuracy as a function of training-set size.

Run from the project root:   python notebooks/04_modeling.py
Outputs printed metrics and four figures into the figures/ folder.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------------
# Paths (work no matter where the script is run from)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "raw" / "Dreher_and_Doyle_input_data.xlsx"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
COMPONENTS = ["Ligand", "Additive", "Base", "Aryl halide"]
N_BITS = 512
RADIUS = 2
N_TREES = 200          # lower to 100 for a faster run, results barely change
SEED = 42

# ---------------------------------------------------------------------------
# Featurisation: one Morgan fingerprint per unique molecule, then concatenate
# ---------------------------------------------------------------------------
df = pd.read_excel(DATA, sheet_name="FullCV_01")
generator = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)

def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return np.array(generator.GetFingerprint(mol))

cache = {}
for col in COMPONENTS:
    for smiles in df[col].unique():
        cache.setdefault(smiles, smiles_to_fingerprint(smiles))

blocks = []
for col in COMPONENTS:
    column_fingerprints = [cache[smiles] for smiles in df[col]]
    blocks.append(np.stack(column_fingerprints))
X = np.concatenate(blocks, axis=1)
y = df["Output"].values
groups = df["Additive"].values          # used to define the out-of-sample split

# Two extra feature sets for the control experiment
X_onehot = pd.get_dummies(df[COMPONENTS].astype(str)).values.astype(float)
X_random = np.random.RandomState(0).randint(0, 2, size=X.shape)

print(f"Feature matrix X: {X.shape}   target y: {y.shape}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_model():
    return RandomForestRegressor(n_estimators=N_TREES, n_jobs=-1, random_state=SEED)

def metrics(y_true, y_pred):
    return (r2_score(y_true, y_pred),
            np.sqrt(mean_squared_error(y_true, y_pred)),
            mean_absolute_error(y_true, y_pred))

idx = np.arange(len(y))
train_r, test_r = train_test_split(idx, test_size=0.3, random_state=SEED)
train_o, test_o = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                         random_state=SEED).split(X, y, groups))

# ---------------------------------------------------------------------------
# 1 & 2. Baseline on both splits, with a side-by-side parity figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, (name, tr, te) in zip(
        axes,
        [("Random split", train_r, test_r),
         ("Held-out additives", train_o, test_o)]):
    model = make_model().fit(X[tr], y[tr])
    pred = model.predict(X[te])
    r2, rmse, mae = metrics(y[te], pred)
    print(f"Baseline | {name:20s} R2={r2:.3f}  RMSE={rmse:.1f}  MAE={mae:.1f}")
    ax.scatter(y[te], pred, s=8, alpha=0.4, edgecolor="none")
    ax.plot([0, 100], [0, 100], "k--", lw=1)
    ax.set_xlabel("Observed yield (%)")
    ax.set_ylabel("Predicted yield (%)")
    ax.set_title(f"{name}\nR2 = {r2:.3f}")
fig.tight_layout()
fig.savefig(FIG_DIR / "parity_random_vs_oos.png", dpi=150)

# ---------------------------------------------------------------------------
# 3. Feature controls: fingerprints vs one-hot vs random, on both splits
# ---------------------------------------------------------------------------
control_results = {}
print("\nFeature controls (R2):")
for name, feat in [("fingerprint", X), ("one-hot", X_onehot), ("random", X_random)]:
    r2_rand = r2_score(y[test_r], make_model().fit(feat[train_r], y[train_r]).predict(feat[test_r]))
    r2_oos = r2_score(y[test_o], make_model().fit(feat[train_o], y[train_o]).predict(feat[test_o]))
    control_results[name] = (r2_rand, r2_oos)
    print(f"  {name:11s} random={r2_rand:.3f}  additive-OOS={r2_oos:.3f}")

labels = list(control_results.keys())
rand_vals = [control_results[k][0] for k in labels]
oos_vals = [control_results[k][1] for k in labels]
xpos = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(xpos - 0.2, rand_vals, width=0.4, label="random split")
ax.bar(xpos + 0.2, oos_vals, width=0.4, label="held-out additives")
ax.set_xticks(xpos); ax.set_xticklabels(labels)
ax.set_ylabel("Test R2"); ax.set_title("Feature controls")
ax.axhline(0, color="black", lw=0.8); ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "feature_controls.png", dpi=150)

# ---------------------------------------------------------------------------
# 4. Split-conformal 90% intervals, coverage on both splits
# ---------------------------------------------------------------------------
print("\nConformal 90% prediction intervals:")
cov_results = {}
for name, tr, te in [("random", train_r, test_r), ("additive-OOS", train_o, test_o)]:
    fit_idx, cal_idx = train_test_split(tr, test_size=0.3, random_state=1)
    model = make_model().fit(X[fit_idx], y[fit_idx])
    residuals = np.abs(y[cal_idx] - model.predict(X[cal_idx]))
    n = len(residuals)
    q = np.quantile(residuals, np.ceil((n + 1) * 0.9) / n, method="higher")
    coverage = np.mean(np.abs(y[te] - model.predict(X[te])) <= q)
    cov_results[name] = coverage
    print(f"  {name:13s} half-width=+/-{q:.1f}%  empirical coverage={coverage:.1%} (target 90%)")

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(list(cov_results.keys()), [v * 100 for v in cov_results.values()])
ax.axhline(90, color="red", ls="--", lw=1.2, label="target 90%")
ax.set_ylabel("Empirical coverage (%)"); ax.set_title("Conformal interval coverage")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "conformal_coverage.png", dpi=150)

# ---------------------------------------------------------------------------
# 5. Learning curve on the random split (fixed test set)
# ---------------------------------------------------------------------------
print("\nLearning curve (random split):")
fractions = [0.05, 0.1, 0.25, 0.5, 1.0]
curve = []
for frac in fractions:
    k = int(len(train_r) * frac)
    subset = np.random.RandomState(0).permutation(train_r)[:k]
    pred = make_model().fit(X[subset], y[subset]).predict(X[test_r])
    r2 = r2_score(y[test_r], pred)
    curve.append(r2)
    print(f"  {int(frac*100):3d}% train ({k:4d} rxns): R2={r2:.3f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([f * 100 for f in fractions], curve, "o-")
ax.set_xlabel("% of training data"); ax.set_ylabel("Test R2")
ax.set_title("Learning curve (random split)")
fig.tight_layout()
fig.savefig(FIG_DIR / "learning_curve.png", dpi=150)

print(f"\nDone. Figures saved to {FIG_DIR}")
