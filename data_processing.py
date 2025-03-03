import pandas as pd

def load_and_clean_data(filepath="anemia_data.xlsx"):
    # Load dataset
    df = pd.read_excel(filepath)

    # Drop duplicates
    df = df.drop_duplicates()

    # Handle missing values (fill with median or drop rows)
    df = df.fillna(df.median(numeric_only=True))
    
    print(df.columns.tolist())
    
    # Filter dataset to include only rows where BMI is between 15 and 35
    df = df[(df["Body Mass Index (BMI) (calculated)"] >= 15) & (df["Body Mass Index (BMI) (calculated)"] <= 35)]
    
    df["Weight of Child (in kg)"] = pd.to_numeric(df["Weight of Child (in kg)"], errors='coerce')
    
    df["Height of Child (in cm)"] = pd.to_numeric(df["Height of Child (in cm)"], errors='coerce')
    
    df["Body Mass Index (BMI) (calculated)"] = pd.to_numeric(df["Body Mass Index (BMI) (calculated)"], errors='coerce')
    
    df["Age of Child (months)"] = pd.to_numeric(df["Age of Child (months)"], errors='coerce')
    
    df["Hemoglobin Level (Hb) (g/dL) by Non-Invasive Device"] = pd.to_numeric(df["Hemoglobin Level (Hb) (g/dL) by Non-Invasive Device"], errors='coerce')
    
    df["Number of People in the Household:"] = pd.to_numeric(df["Number of People in the Household:"], errors='coerce')

    # Convert categorical columns if necessary
    gender = {'Male': 0, 'Female': 1}
    
    education = {"Tertiary education": 0, "Primary education": 1, "Secondary education": 2, "No formal education": 3}

    occupation = { "Private sector employee": 0, "Self-employed": 1, "Unemployed": 2, "Government employee": 3, "Others": 4}

    income = { ">250,000": 0, ">200,000 - 250,000": 1, ">100,000 - 150,000": 2, ">50,000 - 100,000": 3, ">25,000 - 50,000": 4, "<25,000": 5 }
    
    religion = { "Christian": 0, "Muslim": 1, "Other": 2 }
    
    tribe = { "Hausa": 0, "Igbo": 1, "Yoruba": 2 }
    
    marital = { "Married": 0, "Single": 1, "Widowed / Widower": 2, "Divorced": 3, "Separated": 4 }
    
    option = { "Yes": 0, "No": 1 }
    
    howOften = { "Daily": 0, "Weekly": 1, "Rarely": 2, "N/A": 3, "0 - 6 months": 4, "Never": 5 }
    
    dwelling = { "> / = 4 Bedroom house": 0, "2 Bedroom flat": 1, "One Bedroom self-contained": 2, "Others": 3 }

    sanitation = { "Flush toilet": 0, "Pit latrine": 1, "None": 2 }

    source = { "Borehole": 0, "Tap water": 1, "Well": 2 }

    refuse = { "Daily": 0, "Weekly": 1, "Twice Weekly": 2 }
    
    
    df["Parent/Guardian's Education Level"] = df["Parent/Guardian's Education Level"].map(education)
    
    df['Gender'] = df['Gender'].map(gender)
    
    df["Parent/Guardian's Occupation"] = df["Parent/Guardian's Occupation"].map(occupation)
    
    df["Approximate Combined Monthly Family Income (Naira)"] = df["Approximate Combined Monthly Family Income (Naira)"].map(income)
    
    df["Parent/Guardian's Religion"] = df["Parent/Guardian's Religion"].map(religion)
    
    df["Parent/Guardian's Tribe"] = df["Parent/Guardian's Tribe"].map(tribe)
    
    df["Marital Status of Informant"] = df["Marital Status of Informant"].map(marital)
    
    df["Malaria"] = df["Malaria"].map(option)
    
    df["Respiratory infection"] = df["Respiratory infection"].map(option)
    
    df["Diarrhea"] = df["Diarrhea"].map(option)
    
    df["How often does the child eat food rich in Protein (e.g meat, eggs, beans)"] = df["How often does the child eat food rich in Protein (e.g meat, eggs, beans)"].map(howOften)
    
    df["How often does the child eat food that contains Green leaf vegetables?"] = df["How often does the child eat food that contains Green leaf vegetables?"].map(howOften)
    
    df["How often does the child eat fruits"] = df["How often does the child eat fruits"].map(howOften)
    
    df["How often does the child eat Iron-fortified foods?"] = df["How often does the child eat Iron-fortified foods?"].map(howOften)
    
    df["Does your child take Vitamin Supplements?"] = df["Does your child take Vitamin Supplements?"].map(option)
    
    df["Was the child exclusively breastfed?"] = df["Was the child exclusively breastfed?"].map(option)
    
    df["Has the child been dewormed before?"] = df["Has the child been dewormed before?"].map(option)
    
    df["Family Dwelling"] = df["Family Dwelling"].map(dwelling)
    
    df["Sanitation Facilities in Household"] = df["Sanitation Facilities in Household"].map(sanitation)
    
    df["Water Source for Household"] = df["Water Source for Household"].map(source)
    
    df["Does the child regularly sleep under an insecticide-treated mosquito net?"] = df["Does the child regularly sleep under an insecticide-treated mosquito net?"].map(option)
    
    df["Refuse Disposal Frequency"] = df["Refuse Disposal Frequency"].map(refuse)    

    # Feature selection (optional)
    selected_features = ["Age of Child (months)",
        "Parent/Guardian's Education Level",
        "Parent/Guardian's Occupation",
        "Approximate Combined Monthly Family Income (Naira)",
        "Parent/Guardian's Religion",
        "Parent/Guardian's Tribe",
        "Marital Status of Informant",
        "Malaria",
        "Respiratory infection",
        "Diarrhea",
        "How often does the child eat food rich in Protein (e.g meat, eggs, beans)",
        "How often does the child eat food that contains Green leaf vegetables?",
        "How often does the child eat fruits",
        "How often does the child eat Iron-fortified foods?",
        "Does your child take Vitamin Supplements?",
        "Was the child exclusively breastfed?",
        "Has the child been dewormed before?",
        "Family Dwelling",
        "Number of People in the Household:",
        "Sanitation Facilities in Household",
        "Number of Toilets",
        "Water Source for Household",
        "Does the child regularly sleep under an insecticide-treated mosquito net?",
        "Refuse Disposal Frequency",
        "Weight of child (in kg)", 
        "Height of child (in cm)",
        "Body Mass Index (BMI) (calculated)",
        "Hemoglobin Level (Hb) (g/dL) by Non-Invasive Device"]
    if set(selected_features).issubset(df.columns):
        df = df[selected_features]

    return df

# Test function
if __name__ == "__main__":
    df = load_and_clean_data()
    print(df.head())