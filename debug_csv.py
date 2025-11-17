import pandas as pd

try:
    # Read only the header row to get column names
    df_header = pd.read_csv('crimes.csv', nrows=0)
    print("Columns found in crimes.csv:")
    print(df_header.columns.tolist())

    # Read the first 5 rows to inspect the data
    df_sample = pd.read_csv('crimes.csv', nrows=5)
    print("\nSample data from crimes.csv:")
    print(df_sample.to_string())

except Exception as e:
    print(f"An error occurred: {e}")
