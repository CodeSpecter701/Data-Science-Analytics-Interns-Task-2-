import pandas as pd
from pathlib import Path

# Create dummy data
data = {
    "age": [25, 32, 47, 51, 29],
    "job": ["admin.", "technician", "management", "services", "student"],
    "marital": ["single", "married", "married", "divorced", "single"],
    "education": ["bachelor", "diploma", "master", "high.school", "bachelor"],
    "balance": [1200, 3400, 15000, 2300, 500],
    "housing": ["yes", "yes", "no", "yes", "no"],
    "loan": ["no", "yes", "no", "no", "no"],
    "contact": ["cellular", "telephone", "cellular", "cellular", "telephone"],
    "day": [5, 12, 18, 22, 30],
    "month": ["may", "jun", "jul", "aug", "sep"],
    "duration": [120, 340, 220, 150, 90],
    "campaign": [1, 2, 1, 3, 1],
    "pdays": [-1, 100, -1, 200, -1],
    "previous": [0, 1, 0, 2, 0],
    "deposit": ["yes", "no", "yes", "no", "yes"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Create data folder
data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

# Save CSV with ; separator
csv_path = data_folder / "bank.csv"
df.to_csv(csv_path, sep=';', index=False)

print(f"CSV file created successfully at: {csv_path}")