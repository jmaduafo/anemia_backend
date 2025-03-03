import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load trained model
with open("anemia_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define FastAPI app
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define request body model
class AnemiaInput(BaseModel):
    Age: int
    # String
    Education: int 
    # String
    Income: int
    # String
    Occupation: int
    # String
    Religion: int
    # String
    Tribe: int
    # String
    MaritalStatus:int 
    # String
    Malaria: int
    # String
    RespiratoryInfection: int
    # String
    Diarrhea: int
    # String
    Protein: int
    # String
    GreenLeaf: int
    # String
    Fruits: int
    # String
    IronFortified: int
    # String
    Breastfed:  int
    # String
    Vitamins: int
    # String
    Dewormed: int
    # String
    Dwelling: int
    HouseholdNum: int
    # String
    Sanitation: int
    ToiletNum: int
    # String
    WaterSource: int
    # String
    MosquitoNet: int
    # String
    Refuse:  int
    Weight: float
    Height: float
    Bmi: float
    # String
    Gender:  int
    Hemoglobin: float


# Prediction endpoint
@app.post("/predict")
def predict_anemia(data: AnemiaInput):
    # Define mappings for categorical variables
    education_mapping = {"Tertiary education": 0, "Primary education": 1, "Secondary education": 2, "No formal education": 3}
    occupation_mapping = {"Private sector employee": 0, "Self-employed": 1, "Unemployed": 2, "Government employee": 3, "Others": 4}
    income_mapping = {">250,000": 0, ">200,000 - 250,000": 1, ">100,000 - 150,000": 2, ">50,000 - 100,000": 3, ">25,000 - 50,000": 4, "<25,000": 5}
    religion_mapping = {"Christian": 0, "Muslim": 1, "Other": 2}
    tribe_mapping = {"Hausa": 0, "Igbo": 1, "Yoruba": 2}
    marital_mapping = {"Married": 0, "Single": 1, "Widowed / Widower": 2, "Divorced": 3, "Separated": 4}
    option_mapping = {"Yes": 0, "No": 1}
    howOften_mapping = {"Daily": 0, "Weekly": 1, "Rarely": 2, "N/A": 3, "0 - 6 months": 4, "Never": 5}
    dwelling_mapping = {"> / = 4 Bedroom house": 0, "2 Bedroom flat": 1, "One Bedroom self-contained": 2, "Others": 3}
    sanitation_mapping = {"Flush toilet": 0, "Pit latrine": 1, "None": 2}
    source_mapping = {"Borehole": 0, "Tap water": 1, "Well": 2}
    refuse_mapping = {"Daily": 0, "Weekly": 1, "Twice Weekly": 2}


    input_data = pd.DataFrame([data.dict()])
    
    # Rename input columns to match model's expected names
    input_data = input_data.rename(columns={
        "Age": "Age of Child (months)",
        "Education": "Parent/Guardian's Education Level",
        "Occupation": "Parent/Guardian's Occupation",
        "Income": "Approximate Combined Monthly Family Income (Naira)", 
        "Religion": "Parent/Guardian's Religion",
        "Tribe": "Parent/Guardian's Tribe",
        "MaritalStatus": "Marital Status of Informant",
        "RespiratoryInfection": "Respiratory infection",
        "Protein": "How often does the child eat food rich in Protein (e.g meat, eggs, beans)",
        "GreenLeaf": "How often does the child eat food that contains Green leaf vegetables?",
        "Fruits": "How often does the child eat fruits",
        "IronFortified": "How often does the child eat Iron-fortified foods?",
        "Breastfed": "Was the child exclusively breastfed?",
        "Vitamins": "Does your child take Vitamin Supplements?",
        "Dewormed": "Has the child been dewormed before?",
        "Dwelling": "Family Dwelling",
        "HouseholdNum": "Number of People in the Household:",
        "Sanitation": "Sanitation Facilities in Household",
        "ToiletNum": "Number of Toilets",
        "WaterSource": "Water Source for Household",
        "MosquitoNet": "Does the child regularly sleep under an insecticide-treated mosquito net?",
        "Refuse": "Refuse Disposal Frequency",
        "Weight": "Weight of Child (in kg)",
        "Height": "Height of Child (in cm)",
        "Bmi": "Body Mass Index (BMI) (calculated)",
        "Hemoglobin": "Hemoglobin Level (Hb) (g/dL) by Non-Invasive Device",    
    })
    
    # "Classification of Anemia Status with Non-Invasive Device",
    # "Anemia Status with Non-Invasive Device",
    
    # Apply encoding
    # input_data["Parent/Guardian's Education Level"] = input_data["Parent/Guardian's Education Level"].map(education_mapping)
    # input_data["Parent/Guardian's Occupation"] = input_data["Parent/Guardian's Occupation"].map(occupation_mapping)
    # input_data["Approximate Combined Monthly Family Income (Naira)"] = input_data["Approximate Combined Monthly Family Income (Naira)"].map(income_mapping)
    # input_data["Parent/Guardian's Religion"] = input_data["Parent/Guardian's Religion"].map(religion_mapping)
    # input_data["Parent/Guardian's Tribe"] = input_data["Parent/Guardian's Tribe"].map(tribe_mapping)
    # input_data["Marital Status of Informant"] = input_data["Marital Status of Informant"].map(marital_mapping)
    # input_data["Malaria"] = input_data["Malaria"].map(option_mapping)
    # input_data["Respiratory infection"] = input_data["Respiratory infection"].map(option_mapping)
    # input_data["Diarrhea"] = input_data["Diarrhea"].map(option_mapping)
    # input_data["How often does the child eat food rich in Protein (e.g meat, eggs, beans)"] = input_data["How often does the child eat food rich in Protein (e.g meat, eggs, beans)"].map(howOften_mapping)
    # input_data["How often does the child eat food that contains Green leaf vegetables?"] = input_data["How often does the child eat food that contains Green leaf vegetables?"].map(howOften_mapping)
    # input_data["How often does the child eat fruits"] = input_data["How often does the child eat fruits"].map(howOften_mapping)
    # input_data["How often does the child eat Iron-fortified foods?"] = input_data["How often does the child eat Iron-fortified foods?"].map(howOften_mapping)
    # input_data["Does your child take Vitamin Supplements?"] = input_data["Does your child take Vitamin Supplements?"].map(option_mapping)
    # input_data["Was the child exclusively breastfed?"] = input_data["Was the child exclusively breastfed?"].map(option_mapping)
    # input_data["Has the child been dewormed before?"] = input_data["Has the child been dewormed before?"].map(option_mapping)
    # input_data["Family Dwelling"] = input_data["Family Dwelling"].map(dwelling_mapping)
    # input_data["Sanitation Facilities in Household"] = input_data["Sanitation Facilities in Household"].map(sanitation_mapping)
    # input_data["Water Source for Household"] = input_data["Water Source for Household"].map(source_mapping)
    # input_data["Does the child regularly sleep under an insecticide-treated mosquito net?"] = input_data["Does the child regularly sleep under an insecticide-treated mosquito net?"].map(option_mapping)
    # input_data["Refuse Disposal Frequency"] = input_data["Refuse Disposal Frequency"].map(refuse_mapping)
    
     # Check for missing values after encoding
    if input_data.isnull().values.any():
        return {"error": "Some input values could not be mapped. Please check your input."}

    # Ensure only expected columns are passed to the model
    expected_columns = [
        "Age of Child (months)",
        "Gender",
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
        "Weight of Child (in kg)", 
        "Height of Child (in cm)",
        "Body Mass Index (BMI) (calculated)",
        "Hemoglobin Level (Hb) (g/dL) by Non-Invasive Device",
    ]

    input_data = input_data[expected_columns]

    # Make prediction
    prediction = model.predict(input_data)[0]
    return {"anemia": bool(prediction)}

# Run with: python -m uvicorn server:app --reload
