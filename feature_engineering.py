import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(train_data_path, test_data_path):
    # Load train and test datasets
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

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
