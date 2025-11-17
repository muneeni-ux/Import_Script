import pandas as pd

# Assume your files are saved in the same directory as the script.
# You may need to rename the files to 'Main.csv' and 'Revenue.csv'
# or update the file names below.
try:
    df_main = pd.read_csv("MovieDB.xlsx - Main.csv")
    df_revenue = pd.read_csv("MovieDB.xlsx - Revenue.csv")
except FileNotFoundError:
    print("Error: Please ensure the two original CSV files are in the correct directory.")
    # Exit the function if files are not found
    exit() 

# 1. Clean the main dataframe to remove supplementary rows
# Filter for rows that are likely the main movie entry
df_main_cleaned = df_main[(df_main['Movie Name'].notna()) & (df_main['Budget'] > 0)].copy()

# 2. Rename the Revenue column in the revenue file to avoid collision during merge
df_revenue.rename(columns={'Revenue': 'BoxOfficeRevenue_RevFile'}, inplace=True)

# 3. Merge the Main and Revenue dataframes on 'Movie Name'
df_merged = pd.merge(
    df_main_cleaned,
    df_revenue[['Movie Name', 'BoxOfficeRevenue_RevFile']],
    on='Movie Name',
    how='left'
)

# 4. Save the final merged DataFrame to a new CSV file
output_filename = "MovieDB_Merged_Cleaned_Catalog.csv"
df_merged.to_csv(output_filename, index=False)

print(f"\nSuccess! The cleaned and merged file has been saved as {output_filename}")