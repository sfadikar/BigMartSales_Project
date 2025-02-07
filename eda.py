import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_data

def perform_eda():
    """Perform EDA and save visualizations."""
    train_df, _ = load_data()
    
    print(f"Dataset Shape: {train_df.shape}")
    print(f"Dataset Info: {train_df.info()}")
    print(f"Missing Values: {train_df.isnull().sum()}")
    
    # Visualization Example
    plt.figure(figsize=(10, 5))
    sns.histplot(train_df['Item_Outlet_Sales'], kde=True)
    plt.title("Item Outlet Sales Distribution")
    plt.savefig("output/sales_distribution.png")
    plt.show()
