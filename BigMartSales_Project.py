import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


def load_data():
    train_data_path = "data/train_v9rqX0R.csv"
    test_data_path = "data/test_AbJTz2l.csv"
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    return train_data, test_data


def perform_eda(train_data):
    """Perform EDA and save visualizations."""
    print(f"Dataset Shape: {train_data.shape}")
    print(f"Dataset Info: {train_data.info()}")
    print(f"Missing Values: {train_data.isnull().sum()}")

    # Visualization Example
    plt.figure(figsize=(10, 5))
    sns.histplot(train_data['Item_Outlet_Sales'], kde=True)
    plt.title("Item Outlet Sales Distribution")
    plt.savefig("output/sales_distribution.png")
    plt.show()


def preprocess_data(train_data, test_data):
    # Handle missing values
    for dataset in [train_data, test_data]:
        dataset.fillna({'Item_Weight': dataset['Item_Weight'].mean(), 'Outlet_Size': 'Medium'}, inplace=True)

    # Split train data into training and validation datasets
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Encode categorical features
    categorical_columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
                            'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    # Fit on training and transform all datasets
    train_encoded = pd.DataFrame(encoder.fit_transform(train_data[categorical_columns]))
    val_encoded = pd.DataFrame(encoder.transform(val_data[categorical_columns]))
    test_encoded = pd.DataFrame(encoder.transform(test_data[categorical_columns]))

    # Assign encoded feature names
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    train_encoded.columns = encoded_columns
    val_encoded.columns = encoded_columns
    test_encoded.columns = encoded_columns

    # Drop original columns and concatenate encoded data
    train_data = train_data.drop(columns=categorical_columns)
    val_data = val_data.drop(columns=categorical_columns)
    test_data = test_data.drop(columns=categorical_columns)

    train_preprocessed = pd.concat([train_data.reset_index(drop=True), train_encoded.reset_index(drop=True)], axis=1)
    val_preprocessed = pd.concat([val_data.reset_index(drop=True), val_encoded.reset_index(drop=True)], axis=1)
    test_preprocessed = pd.concat([test_data.reset_index(drop=True), test_encoded.reset_index(drop=True)], axis=1)

    # Save preprocessed data
    train_preprocessed.to_csv('data/preprocessed_train_data.csv', index=False)
    val_preprocessed.to_csv('data/processed_validation_data.csv', index=False)
    test_preprocessed.to_csv('data/preprocessed_test_data.csv', index=False)

    # Save encoder for future use
    with open('output/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

    print("Preprocessed train, validation, and test data saved successfully.")


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


def evaluate_model():
    # Load preprocessed test data
    X_test = pd.read_csv("data/preprocessed_test_data.csv")

    # Load the original test data to retrieve missing columns if necessary
    test_data = pd.read_csv("data/test_AbJTz2l.csv")

    # Load feature names to ensure proper alignment
    with open('output/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    # Ensure the test data has only the required features
    X_test_features = X_test.reindex(columns=feature_names, fill_value=0)

    # Load the model
    model = joblib.load("output/final_model.pkl")

    # Predict sales
    predictions = model.predict(X_test_features)

    # Add the missing 'Outlet_Identifier' if it doesn't exist in X_test
    if "Outlet_Identifier" not in X_test.columns:
        X_test["Outlet_Identifier"] = test_data["Outlet_Identifier"]

    # Save predictions for review
    X_test['Item_Outlet_Sales'] = predictions
    submission = X_test[["Item_Identifier", "Outlet_Identifier", "Item_Outlet_Sales"]]
    submission.to_csv('output/submission.csv', index=False)

    print("Model evaluation complete. Predictions saved at output/submission.csv.")


def main():
    print("Starting Big Mart Sales Project Pipeline...")

    # Load data
    train_data, test_data = load_data()

    # Step 1: Perform EDA
    perform_eda(train_data)

    # Step 2: Preprocess Data
    preprocess_data(train_data, test_data)

    # Step 3: Train Model
    train_model()

    # Step 4: Evaluate Model
    evaluate_model()

    print("Pipeline executed successfully.")


if __name__ == "__main__":
    main()
