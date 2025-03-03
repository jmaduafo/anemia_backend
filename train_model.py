import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from data_processing import load_and_clean_data

def train_and_save_model():
    # Load cleaned dataset
    df = load_and_clean_data()

    # Convert categorical columns to numeric
    categorical_columns = df.select_dtypes(include=['object']).columns
    print("Categorical Columns:", categorical_columns)

    # Apply Label Encoding (convert text to numbers)
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Convert to string before encoding
        label_encoders[col] = le

    # Define features (X) and target variable (y)
    X = df[["Age of Child (months)", 
            "Gender", "Parent/Guardian's Education Level", "Parent/Guardian's Occupation", "Approximate Combined Monthly Family Income (Naira)", "Parent/Guardian's Religion", "Parent/Guardian's Tribe", "Marital Status of Informant","Malaria", "Respiratory infection", "Diarrhea", "How often does the child eat food rich in Protein (e.g meat, eggs, beans)", "How often does the child eat food that contains Green leaf vegetables?", "How often does the child eat fruits",
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
        ]]  # Keep this column!
   
   # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Encode the target variable
    y = label_encoder.fit_transform(df["Classification of Anemia Status with Non-Invasive Device"])

    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save trained model
    with open("anemia_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved as anemia_model.pkl")

# Train and save model
if __name__ == "__main__":
    train_and_save_model()