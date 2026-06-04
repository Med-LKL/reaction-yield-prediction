# Reaction-Yield Prediction with Honest Evaluation

Predicting Buchwald-Hartwig C-N coupling yields from molecular structure, with an
emphasis on honest validation: what the model really learns, where it breaks, and
how much its uncertainty estimates can be trusted.

## Background

Reaction-yield prediction is a useful tool for prioritising experiments and reducing
wasted lab effort. This project uses the Buchwald-Hartwig dataset of Ahneman et al.
(Science, 2018), a high-throughput screen of 3,955 Pd-catalysed C-N couplings built
from 4 ligands, 22 isoxazole additives, 3 bases, and 15 aryl halides, where many of
the additives deliberately poison the reaction. The measured yields are right-skewed
with a large fraction near zero, reflecting those genuine failures.

The goal here is not only to predict yield, but to stress-test the model the way an
industrial scientist would before trusting it on new chemistry.

## Data and features

Each reaction is four molecules (ligand, additive, base, aryl halide), supplied as
SMILES. Because there are only 44 unique building blocks across all 3,955 reactions,
each is fingerprinted once and reused. The baseline representation is a 512-bit Morgan
fingerprint (radius 2) per component, concatenated into a 2,048-dimensional vector per
reaction so that the four chemical roles stay separable.

## Method

A random forest regressor is trained and evaluated under two splitting schemes:

- Random split: reactions are shuffled and split 70/30. This is the standard benchmark.
- Held-out additives: whole additives are removed from training, so the test set
  contains chemistry the model has never seen. This is the realistic generalisation test.

Three further experiments probe how much to trust the model: feature controls, conformal
prediction intervals, and a learning curve.

## Results

| Result | Random split | Held-out additives |
|---|---|---|
| Baseline RF (fingerprints), R2 | 0.929 | 0.774 |
| One-hot identities only, R2 | 0.890 | 0.483 |
| Random features (control), R2 | ~0.00 | ~0.00 |
| Conformal 90% interval coverage | 91.8% | 74.0% |

Learning curve (random split): R2 reaches 0.61 with only 138 training reactions, 0.84
with 692, and 0.93 with the full training set.

What these say:

1. The baseline reproduces the published benchmark (about R2 0.92 on the random split),
   confirming the pipeline is correct.
2. One-hot identity features nearly match structural fingerprints on the random split but
   collapse when additives are held out, because identity tags carry no information about
   a molecule never seen in training. This reproduces the well-known Chuang and Keiser
   critique (Science, 2018): much of the easy accuracy is memorising which components are
   present rather than learning transferable chemistry. Only structural features generalise.
3. Random features score near zero on both splits, confirming there is no data leakage.
4. Conformal 90% intervals achieve 91.8% coverage on the random split (the guarantee
   holds) but only 74% out-of-sample. This is expected: conformal coverage relies on the
   test data resembling the calibration data, an assumption that breaks under the
   distribution shift introduced by holding out additives.
5. The model learns usefully from limited data, which matters when each experiment is
   expensive.

## Honest limitations

- The headline R2 on the random split overstates real-world performance; the held-out
  additive number is the more honest estimate of generalisation to new chemistry.
- Morgan fingerprints encode structure, not electronics. They are a fast baseline, not a
  mechanistically grounded representation.
- The conformal guarantee is only marginal and degrades under distribution shift, as the
  coverage result shows directly.

## Future work

A natural extension is to replace structural fingerprints with quantum-chemical
descriptors (computed once for the 44 unique molecules), which should generalise better
out-of-sample. A second direction is a delta-learning surrogate that predicts
DFT-quality reaction energetics from a cheap semiempirical baseline, with the conformal
uncertainty used to decide when a prediction is trustworthy and when to fall back to a
full DFT calculation. That pattern is directly relevant to physics-based reaction- and
stability-prediction pipelines that combine semiempirical exploration with selective DFT
refinement.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download the data into data/raw/ (see notebooks/01_load_data.py)
python notebooks/04_modeling.py
```

Figures are written to `figures/`.

## References

- D. T. Ahneman, J. G. Estrada, S. Lin, S. D. Dreher, A. G. Doyle. Predicting reaction
  performance in C-N cross-coupling using machine learning. Science 360, 186-190 (2018).
- K. V. Chuang, M. J. Keiser. Comment on "Predicting reaction performance in C-N
  cross-coupling using machine learning". Science 362 (2018).
