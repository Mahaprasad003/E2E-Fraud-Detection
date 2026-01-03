import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        if "Time" in data.columns:
            data = data.drop(columns=["Time"])
            print("Dropped 'Time' column from the dataset.")
        
        X = data.drop(columns=["Class"])
        y = data["Class"]
        print("Separated features and target variable.")
        return X, y
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        raise