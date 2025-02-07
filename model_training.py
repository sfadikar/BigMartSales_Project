import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import pickle

def train_model():
    # Load preprocessed data
    data = pd.read_csv("data/preprocessed_train_data.csv")
    
    # Separate features and target
    X = data.drop(columns=["Item_Outlet_Sales"])
    y = data["Item_Outlet_Sales"]

    # Select only numeric columns
    X = X.select_dtypes(include=np.number)

    # Handle cases where no numeric features remain (add encoding)
    if X.empty:
        raise ValueError("No numeric features in the dataset. Ensure proper encoding during preprocessing.")
    
    # Save the feature names for evaluation use
    feature_names = X.columns.tolist()
    with open('output/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, "output/final_model.pkl")
    print("Model training complete. Model and feature names saved at output/final_model.pkl and output/feature_names.pkl.")

if __name__ == "__main__":
    train_model()
