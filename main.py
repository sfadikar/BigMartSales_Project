from eda import perform_eda
from feature_engineering import preprocess_data
from model_training import train_model
from model_evaluation import evaluate_model

def main():
    print("Starting Big Mart Sales Project Pipeline...")

    # Paths to the input datasets
    train_data_path = "data/train_v9rqX0R.csv"
    test_data_path = "data/test_AbJTz2l.csv"

    perform_eda()
    preprocess_data(train_data_path, test_data_path)
    train_model()
    evaluate_model()

    print("Pipeline executed successfully.")

if __name__ == "__main__":
    main()
