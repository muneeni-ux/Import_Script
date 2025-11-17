# %% [markdown]
# # Los Angeles Crime Data Analysis
#
# **Objective:** To analyze the "Crime in Los Angeles" dataset to understand the geographical distribution of crime, identify time-based patterns, and answer key questions about crime rates and trends.
#
# %% [markdown]
# ## 1. Data Exploration and Cleaning
#
# In this section, we will load the dataset and perform initial exploratory data analysis (EDA). We'll check for missing values, ensure data types are correct, and clean the data for analysis.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
try:
    df = pd.read_csv('crimes.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: crimes.csv not found. Make sure the file is in the correct directory.")
    exit()


# Initial inspection
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
df.info()

# %% [markdown]
# ### 1.1 Clean Column Names and Dates
#
# We'll start by cleaning up the column names to remove any leading/trailing whitespace. Then, we'll convert the date columns to a proper datetime format, which is essential for time-series analysis.

# %%
# Strip whitespace from column names to prevent errors
df.columns = df.columns.str.strip()
print("Column names cleaned.")

# Convert date columns to datetime objects
df['Date Rptd'] = pd.to_datetime(df['Date Rptd'])
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])

# Extract Year and Month for trend analysis
df['Year'] = df['DATE OCC'].dt.year
df['Month'] = df['DATE OCC'].dt.month

print("\nDataset Information after cleaning dates:")
df.info()


# %% [markdown]
# ### 1.2 Handle Missing Geospatial Data
#
# The `LAT` and `LON` columns are missing. We will attempt to extract them from the `LOCATION` column.

# %%
print("Inspecting the LOCATION column format:")
print(df['LOCATION'].head().to_string())

# Extract LAT and LON from the LOCATION column.
# It appears the format is a string like '(34.0522, -118.2437)'.
# We will parse this to create new numeric columns.
if 'LOCATION' in df.columns:
    location_split = df['LOCATION'].str.strip('()').str.split(',', expand=True)

    if location_split.shape[1] == 2:
        df['LAT'] = pd.to_numeric(location_split[0], errors='coerce')
        df['LON'] = pd.to_numeric(location_split[1], errors='coerce')
        print("\nSuccessfully extracted LAT and LON from LOCATION column.")

        print(f"\nNumber of rows before dropping missing geo-data: {len(df)}")
        # Drop rows where the new LAT or LON columns have missing values
        df.dropna(subset=['LAT', 'LON'], inplace=True)
        # Also drop rows where coordinates are (0, 0) as this is likely invalid for LA
        df = df[(df['LAT'] != 0) & (df['LON'] != 0)]
        print(f"Number of rows after dropping missing geo-data: {len(df)}")
    else:
        print("\nCould not parse LAT and LON from LOCATION column as expected.")
        print("Geospatial analysis will be skipped.")
        # To prevent the rest of the script from failing, create empty columns
        df['LAT'] = np.nan
        df['LON'] = np.nan
else:
    print("\nLOCATION column not found. Geospatial analysis will be skipped.")
    df['LAT'] = np.nan
    df['LON'] = np.nan


# %% [markdown]
# ## 2. Answering Key Questions
#
# With the data cleaned, we can now focus on answering the core questions of our analysis through visualizations.
#
# ### Which areas have the highest crime rates?

# %%
# Set the style for the plots
sns.set(style="whitegrid")

# Get crime counts by area
area_crime_counts = df['AREA NAME'].value_counts()

print("\nCrime Counts by Area:")
print(area_crime_counts)

# Plotting crime counts by area
plt.figure(figsize=(12, 9))
sns.barplot(y=area_crime_counts.index, x=area_crime_counts.values, orient='h')
plt.title('Total Crime Incidents by Area')
plt.xlabel('Number of Incidents')
plt.ylabel('Area Name')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Has crime increased or decreased over the last year?
#
# To answer this, we will look at the monthly crime counts for the most recent full year of data available in the dataset.

# %%
# Determine the year to analyze
latest_year = df['Year'].max()
# We analyze the year before the latest year to ensure we have a full 12 months of data
analysis_year = latest_year - 1

df_last_year = df[df['Year'] == analysis_year]
monthly_crime_counts = df_last_year.groupby('Month').size()

print(f"\nMonthly Crime Counts for {analysis_year}:")
print(monthly_crime_counts)

# Plotting the monthly crime trend
plt.figure(figsize=(12, 6))
monthly_crime_counts.plot(kind='line', marker='o', linestyle='-')
plt.title(f'Monthly Crime Trend for {analysis_year}')
plt.xlabel('Month')
plt.ylabel('Number of Incidents')
plt.xticks(ticks=np.arange(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Geospatial and Crime Type Analysis
#
# ### 3.1 Geospatial Distribution of All Crimes
#
# Here we plot all crime incidents on a map of Los Angeles to visualize hotspots.

# %%
# Only run geospatial plots if LAT and LON columns were successfully created
if not df['LAT'].isnull().all():
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='LON', y='LAT', data=df, s=1, alpha=0.1, color='red')
    plt.title('Geospatial Distribution of All Crimes in LA')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.show()
else:
    print("\nSkipping geospatial distribution plot as LAT/LON data is unavailable.")


# %% [markdown]
# ### 3.2 Analysis of Top Crime Types
#
# Let's identify the most common types of crime and visualize their distribution.

# %%
top_10_crimes = df['Crm Cd Desc'].value_counts().nlargest(10)

print("\nTop 10 Most Common Crimes:")
print(top_10_crimes)

plt.figure(figsize=(12, 7))
sns.barplot(y=top_10_crimes.index, x=top_10_crimes.values, orient='h')
plt.title('Top 10 Most Common Crimes')
plt.xlabel('Number of Incidents')
plt.ylabel('Crime Description')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.3 Geospatial Plot of a Specific Crime
#
# To get a more focused view, let's plot the locations of the most common crime: **Vehicle Stolen**.

# %%
if not df['LAT'].isnull().all():
    most_common_crime = top_10_crimes.index[0]
    df_specific_crime = df[df['Crm Cd Desc'] == most_common_crime]

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='LON', y='LAT', data=df_specific_crime, s=5, alpha=0.2)
    plt.title(f'Geospatial Distribution of: {most_common_crime}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.show()
else:
    print("\nSkipping specific crime geospatial plot as LAT/LON data is unavailable.")


# %% [markdown]
# ## 4. Summary and Interpretation
#
# Based on the analysis above, we can draw several conclusions:
#
# 1.  **Highest Crime Areas:** The **Central** area has the highest number of reported crime incidents, followed closely by **Southwest** and **77th Street**. These appear to be significant hotspots.
#
# 2.  **Crime Trend:** The monthly trend analysis for the last full year shows fluctuations in crime rates. There are noticeable peaks and troughs, which could be investigated further for seasonality or correlation with external events.
#
# 3.  **Geospatial Patterns:** The map of all incidents clearly outlines the geography of Los Angeles and shows that crime is widespread but more concentrated in specific clusters, aligning with the findings from the "highest crime areas" analysis.
#
# 4.  **Dominant Crime Type:** **Vehicle theft** is by far the most common crime reported in this dataset. The geospatial plot for this specific crime shows it is prevalent across most of the city, rather than being confined to one or two areas.
#
# ### Further Steps
#
# *   **Deeper Dive:** Analyze trends for specific crime types (e.g., has vehicle theft increased in a particular area?).
# *   **Time-of-Day Analysis:** Use the `TIME OCC` column to see if crimes are more common at certain times of the day or night.
# *   **Predictive Modeling:** Build a model to forecast future crime rates for different areas.
# %%