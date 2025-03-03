import pickle
import pandas as pd
from preprocess import clean_data  
from sklearn.preprocessing import StandardScaler

# Load the trained ML model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

def predict_anemia(input_data):
    """
    Function to predict anemia based on input patient data.
    input_data: Dictionary containing patient features
    Returns: Predicted label (Anemic or Not)
    """
    df = pd.DataFrame([input_data])  # Convert input data to DataFrame
    df = clean_data(df)  # Apply preprocessing

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    prediction = model.predict(df_scaled)
    return "Anemic" if prediction[0] == 1 else "Not Anemic"
    