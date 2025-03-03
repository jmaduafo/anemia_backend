import pandas as pd
import joblib
import pickle

# Load trained model
model = joblib.load("anemia_model.pkl")


# Load old test data
test_data = pd.read_excel("anemia_data.xlsx", engine="openpyxl")  # Ensure correct filename

# Define categorical mappings (same as in data_processing.py)
gender = {'Male': 0, 'Female': 1}
education = {"Tertiary education": 0, "Primary education": 1, "Secondary education": 2, "No formal education": 3}
occupation = {"Private sector employee": 0, "Self-employed": 1, "Unemployed": 2, "Government employee": 3, "Others": 4}
income = {">250,000": 0, ">200,000 - 250,000": 1, ">100,000 - 150,000": 2, ">50,000 - 100,000": 3, ">25,000 - 50,000": 4, "<25,000": 5}
religion = {"Christian": 0, "Muslim": 1, "Other": 2}
tribe = {"Hausa": 0, "Igbo": 1, "Yoruba": 2}
marital = {"Married": 0, "Single": 1, "Widowed / Widower": 2, "Divorced": 3, "Separated": 4}
option = {"Yes": 0, "No": 1}
howOften = {"Daily": 0, "Weekly": 1, "Rarely": 2, "N/A": 3, "0 - 6 months": 4, "Never": 5}
dwelling = {"> / = 4 Bedroom house": 0, "2 Bedroom flat": 1, "One Bedroom self-contained": 2, "Others": 3}
sanitation = {"Flush toilet": 0, "Pit latrine": 1, "None": 2}
source = {"Borehole": 0, "Tap water": 1, "Well": 2}
refuse = {"Daily": 0, "Weekly": 1, "Twice Weekly": 2}

# Apply mappings to test data
test_data["Gender"] = test_data["Gender"].map(gender)
test_data["Parent/Guardian's Education Level"] = test_data["Parent/Guardian's Education Level"].map(education)
test_data["Parent/Guardian's Occupation"] = test_data["Parent/Guardian's Occupation"].map(occupation)
test_data["Approximate Combined Monthly Family Income (Naira)"] = test_data["Approximate Combined Monthly Family Income (Naira)"].map(income)
test_data["Parent/Guardian's Religion"] = test_data["Parent/Guardian's Religion"].map(religion)
test_data["Parent/Guardian's Tribe"] = test_data["Parent/Guardian's Tribe"].map(tribe)
test_data["Marital Status of Informant"] = test_data["Marital Status of Informant"].map(marital)
test_data["Malaria"] = test_data["Malaria"].map(option)
test_data["Respiratory infection"] = test_data["Respiratory infection"].map(option)
test_data["Diarrhea"] = test_data["Diarrhea"].map(option)
test_data["How often does the child eat food rich in Protein (e.g meat, eggs, beans)"] = test_data["How often does the child eat food rich in Protein (e.g meat, eggs, beans)"].map(howOften)
test_data["How often does the child eat food that contains Green leaf vegetables?"] = test_data["How often does the child eat food that contains Green leaf vegetables?"].map(howOften)
test_data["How often does the child eat fruits"] = test_data["How often does the child eat fruits"].map(howOften)
test_data["How often does the child eat Iron-fortified foods?"] = test_data["How often does the child eat Iron-fortified foods?"].map(howOften)
test_data["Does your child take Vitamin Supplements?"] = test_data["Does your child take Vitamin Supplements?"].map(option)
test_data["Was the child exclusively breastfed?"] = test_data["Was the child exclusively breastfed?"].map(option)
test_data["Has the child been dewormed before?"] = test_data["Has the child been dewormed before?"].map(option)
test_data["Family Dwelling"] = test_data["Family Dwelling"].map(dwelling)
test_data["Sanitation Facilities in Household"] = test_data["Sanitation Facilities in Household"].map(sanitation)
test_data["Water Source for Household"] = test_data["Water Source for Household"].map(source)
test_data["Does the child regularly sleep under an insecticide-treated mosquito net?"] = test_data["Does the child regularly sleep under an insecticide-treated mosquito net?"].map(option)
test_data["Refuse Disposal Frequency"] = test_data["Refuse Disposal Frequency"].map(refuse)

# Convert numeric columns to appropriate data type
numeric_columns = [
    "Age of Child (months)", "Weight of Child (in kg)", "Height of Child (in cm)",
    "Body Mass Index (BMI) (calculated)", "Hemoglobin Level (Hb) (g/dL) by Non-Invasive Device",
    "Number of People in the Household:", "Number of Toilets"
]
for col in numeric_columns:
    test_data[col] = pd.to_numeric(test_data[col], errors='coerce')

# Define feature columns (must match training data)
feature_columns = [
    "Age of Child (months)", "Gender", "Parent/Guardian's Education Level",
    "Parent/Guardian's Occupation", "Approximate Combined Monthly Family Income (Naira)",
    "Parent/Guardian's Religion", "Parent/Guardian's Tribe", "Marital Status of Informant",
    "Malaria", "Respiratory infection", "Diarrhea",
    "How often does the child eat food rich in Protein (e.g meat, eggs, beans)",
    "How often does the child eat food that contains Green leaf vegetables?",
    "How often does the child eat fruits", "How often does the child eat Iron-fortified foods?",
    "Does your child take Vitamin Supplements?", "Was the child exclusively breastfed?",
    "Has the child been dewormed before?", "Family Dwelling", "Number of People in the Household:",
    "Sanitation Facilities in Household", "Number of Toilets", "Water Source for Household",
    "Does the child regularly sleep under an insecticide-treated mosquito net?",
    "Refuse Disposal Frequency", "Weight of Child (in kg)", "Height of Child (in cm)",
    "Body Mass Index (BMI) (calculated)", "Hemoglobin Level (Hb) (g/dL) by Non-Invasive Device"
]

# Select only relevant columns
X_old_test = test_data[feature_columns]

# Make predictions
test_data["Predicted_Anemia"] = model.predict(X_old_test)
# Attach predictions to test data
anemia_mapping = {0: "No Anemia", 1: "Mild/Moderate Anemia", 2: "Severe Anemia"}
test_data["Predicted_Anemia_Label"] = test_data["Predicted_Anemia"].map(anemia_mapping)

# Save results
test_data.to_excel("old_results_with_predictions.xlsx", index=False)

print("Updated old test results saved as old_results_with_predictions.xlsx")