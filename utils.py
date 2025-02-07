import pandas as pd
import joblib

def load_data():
    """Load the training and test datasets."""
    train_path = 'data/train_v9rqX0R.csv'
    test_path = 'data/test_AbJTz2l.csv'
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print("Data loaded successfully.")
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def save_preprocessed_data(train_df, test_df, encoder):
    """Save preprocessed datasets and the encoder."""
    try:
        train_df.to_csv('output/train_preprocessed.csv', index=False)
        test_df.to_csv('output/test_preprocessed.csv', index=False)
        joblib.dump(encoder, 'output/encoder.pkl')
        print("Preprocessed data and encoder saved successfully.")
    except Exception as e:
        print(f"Error while saving preprocessed data or encoder: {e}")
        raise


def load_preprocessed_data():
    """Load preprocessed datasets and the encoder."""
    try:
        train_df = pd.read_csv('output/train_preprocessed.csv')
        test_df = pd.read_csv('output/test_preprocessed.csv')
        encoder = joblib.load('output/encoder.pkl')
        print("Preprocessed data and encoder loaded successfully.")
        return train_df, test_df, encoder
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def save_model(model):
    """Save the trained model."""
    try:
        joblib.dump(model, 'output/final_model.pkl')
        print("Model saved successfully at 'output/final_model.pkl'.")
    except Exception as e:
        print(f"Error while saving the model: {e}")
        raise


def load_model():
    """Load the trained model."""
    try:
        model = joblib.load('output/final_model.pkl')
        print("Model loaded successfully.")
        return model
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def save_submission_file(predictions, test_df):
    """Save predictions as a CSV submission file."""
    try:
        submission = test_df[['Item_Identifier', 'Outlet_Identifier']].copy()
        submission['Item_Outlet_Sales'] = predictions
        submission.to_csv('output/submission.csv', index=False)
        print("Submission file saved at 'output/submission.csv'.")
    except Exception as e:
        print(f"Error while saving the submission file: {e}")
        raise
