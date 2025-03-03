import os
import pandas as pd

# Load the old and new Excel files
old_df = pd.read_excel("old_results_with_predictions.xlsx")
new_df = pd.read_excel("test_results.xlsx")

# Print column names to check correct naming
print("Old file columns:", old_df.columns.tolist())
print("New file columns:", new_df.columns.tolist())

# Check if "Predicted_Anemia" column exists
pred_col = "Predicted_Anemia"  # Update this if it's named differently

if pred_col not in old_df.columns or pred_col not in new_df.columns:
    raise KeyError(f"Column '{pred_col}' not found in one of the files. Check printed column names.")

# Add a unique ID column if not already present
if "Unique_ID" not in old_df.columns:
    old_df.insert(0, "Unique_ID", range(1, len(old_df) + 1))
if "Unique_ID" not in new_df.columns:
    new_df.insert(0, "Unique_ID", range(1, len(new_df) + 1))

# Ensure both datasets have a common unique identifier for row comparison
if "Unique_ID" in old_df.columns and "Unique_ID" in new_df.columns:
    # Merge the two datasets based on Unique_ID
    comparison_df = old_df.merge(new_df, on="Unique_ID", suffixes=("_Old", "_New"))
    
    # Check if "Predicted_Anemia" column exists, if not, add it
    if "Predicted_Anemia" not in old_df.columns:
        comparison_df["Predicted_Anemia_Old"] = None  # Fill with NaN values

    # Map numerical predictions to readable labels
    anemia_mapping = {0: "No Anemia", 1: "Mild/Moderate Anemia", 2: "Severe Anemia"}
    
    # Convert predictions into readable labels
    comparison_df["Predicted_Anemia_Label_Old"] = comparison_df["Predicted_Anemia_Old"].map(anemia_mapping)
    comparison_df["Predicted_Anemia_Label_New"] = comparison_df["Predicted_Anemia_New"].map(anemia_mapping)

    # Save the comparison results
    comparison_df.to_excel("comparison_results.xlsx", index=False)

    print("Comparison completed! Check 'comparison_results.xlsx'.")
else:
    print("Error: Unique_ID column missing in one or both files.")

print("Comparison file saved as comparison_results.xlsx")

os.system("start excel comparison_results.xlsx")