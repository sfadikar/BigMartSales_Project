import pandas as pd
import joblib
import pickle

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

if __name__ == "__main__":
    evaluate_model()
