import os
import pandas as pd

def save_as_csv(df: pd.DataFrame, filename: str, output_dir: str = 'outputs') -> None:
    """
    Appends a DataFrame to an existing CSV file, or creates it if it doesn't exist.

    Args:
        df (pd.DataFrame): The DataFrame to save or append.
        filename (str): The name of the CSV file (e.g., 'results.csv').
        output_dir (str): The directory where the file will be stored.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    if os.path.exists(path):
        df.to_csv(path, mode='a', header=False, index=False)
        print(f"➕ Appended data to existing CSV at: {path}")
    else:
        df.to_csv(path, mode='w', header=True, index=False)
        print(f"✅ New CSV created at: {path}")