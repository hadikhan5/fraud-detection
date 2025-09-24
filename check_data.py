# Import statements
import pandas as pd
from pathlib import Path

# Connects to data file, ensures existence, then inserts into DF
csv_path = Path("data/raw/creditcard.csv")
assert csv_path.exists(), "creditcard.csv does not exist"
df = pd.read_csv(csv_path)

# Preliminary data checks
print("Shape: ", df.shape)
print("First 10 Columns: ", list(df.columns)[:10], "...")
print ("Class Counts:\n", df["Class"].value_counts())
print("Fraud Rate: ", df["Class"].mean())
print("Missing Values Total: ", df.isna().sum().sum())