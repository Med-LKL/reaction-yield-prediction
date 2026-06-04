from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "raw" / "Dreher_and_Doyle_input_data.xlsx"
FIG_DIR = PROJECT_ROOT / "figures"

df = pd.read_excel(DATA, sheet_name="FullCV_01")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["Output"], bins=30, edgecolor="black")
ax.set_xlabel("Yield (%)")
ax.set_ylabel("Number of reactions")
ax.set_title("Distribution of Buchwald-Hartwig yields (n = 3955)")
fig.tight_layout()
fig.savefig(FIG_DIR / "yield_distribution.png", dpi=150)
plt.show()
