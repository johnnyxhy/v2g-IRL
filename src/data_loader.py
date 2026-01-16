import pandas as pd
import numpy as np

def load_trajectories(file_path):
    """
    Load trajectory data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file containing trajectory data.

    Returns:
    pd.DataFrame: A DataFrame containing the trajectory data.
    """

    try:
        df = pd.read_excel(file_path, sheet_name='Data')
    except ValueError:
        print("Sheet not found. Checking available sheets...")
        xl = pd.ExcelFile(file_path)
        print(f"Available sheets: {xl.sheet_names}")
        raise

    # 
    
# Example usage:
df = load_trajectories("data/EVdataset.xlsx")