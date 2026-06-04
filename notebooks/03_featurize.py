from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "raw" / "Dreher_and_Doyle_input_data.xlsx"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATA, sheet_name="FullCV_01")

COMPONENTS = ["Ligand", "Additive", "Base", "Aryl halide"]
N_BITS = 512
RADIUS = 2

generator = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)

def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return np.array(generator.GetFingerprint(mol))

# Fingerprint each unique molecule once
cache = {}
for col in COMPONENTS:
    for smiles in df[col].unique():
        cache.setdefault(smiles, smiles_to_fingerprint(smiles))
print("unique molecules fingerprinted:", len(cache))

# Build one concatenated row per reaction
blocks = [np.stack([cache[smiles] for smiles in df[col]]) for col in COMPONENTS]
X = np.concatenate(blocks, axis=1)
y = df["Output"].values

print("X shape:", X.shape)
print("y shape:", y.shape)

np.save(PROC_DIR / "X_fingerprints.npy", X)
np.save(PROC_DIR / "y_yields.npy", y)
print("saved features to", PROC_DIR)
