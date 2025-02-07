# BigMartSales_Project

BigMart Sales Prediction Project 🚀
This repository contains a machine learning solution for predicting sales at Big Mart outlets using structured data. The goal is to predict the Item_Outlet_Sales for each combination of items and outlets using various machine learning techniques.

📁 Project Structure
BigMartSales_Project/
├── data/                     # Datasets (raw, preprocessed, and validation data)
│   ├── preprocessed_test_data.csv
│   ├── preprocessed_train_data.csv
│   ├── processed_validation_data.csv
│   ├── test_AbJTz2l.csv
│   └── train_v9rqX0R.csv
├── output/                   # Model outputs and generated predictions
│   ├── encoder.pkl
│   ├── feature_names.pkl
│   ├── final_model.pkl
│   ├── sales_distribution.png
│   └── submission.csv
├── BigMartSales_Project.py   # Combination of all the following 5 python files
├── eda.py                    # Exploratory Data Analysis (EDA) script
├── feature_engineering.py    # Feature Engineering script
├── main.py                   # Main training and model execution script
├── model_evaluation.py       # Model evaluation script
├── model_training.py         # Model training script
├── README.md                 # Project instructions (this file)
├── Rank_Score.png            # Rank & Score of my submission
└── 1_Page_Approach_Note.pdf  # Approach of this case study


⚙️ How to Set Up and Run the Project
1️⃣ Clone the Repository
git clone https://github.com/sfadikar/BigMartSales_Project.git
cd BigMartSales_Project

2️⃣ Install Dependencies
Ensure you have Python 3.7 or higher installed. Install the required packages:
pip install -r requirements.txt

3️⃣ Run the Main Script
To run the entire project pipeline, execute the main script:
```bash
python BigMartSales_Project.py

4️⃣ View the Results
The predictions will be saved at:
output/submission.csv

📊 Project Highlights
Data Preprocessing:
Addressed missing values and handled categorical encoding.

Feature Engineering:
Created meaningful features such as Item_Type_Category, Outlet_Age, and Item_Fat_Content.

EDA Highlights:
Visualized sales distribution, missing data patterns, and key correlations.

Model Experiments:
Evaluated multiple models (Random Forest, Decision Tree).
Selected the best model based on RMSE and leaderboard score.

📈 Results on Leaderboard
Rank Achieved: 4783
Public Score: 1211.4841672167113

📦 Dependencies
pandas
numpy
scikit-learn
joblib

📝 Author
Soumyadip Fadikar
Email: rkmvpsf@gmail.com

