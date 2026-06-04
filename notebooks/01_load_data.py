from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "raw" / "Dreher_and_Doyle_input_data.xlsx"

df = pd.read_excel(DATA, sheet_name="FullCV_01")

print("shape (rows, columns):", df.shape)
print("\ncolumn names:", list(df.columns))
print("\nfirst 3 rows:\n", df.head(3))
print("\nmissing values per column:\n", df.isna().sum())
print("\nyield summary:\n", df["Output"].describe().round(2))
for col in ["Ligand", "Additive", "Base", "Aryl halide"]:
    print(f"unique {col}: {df[col].nunique()}")
