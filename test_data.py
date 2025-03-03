import pandas as pd

# Load the original dataset (replace with your actual dataset file)
df = pd.read_excel("anemia_data.xlsx", engine="openpyxl")

# Randomly sample 200 rows
test_data = df.sample(n=200, random_state=42)  # Random state ensures reproducibility

# Save to CSV
test_data.to_csv("test_data.xlsx", index=False)

print("✅ 200 random data points selected and saved as test_data.csv!")