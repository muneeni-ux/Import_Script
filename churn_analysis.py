# %% [markdown]
# # Customer Churn Prediction Analysis
#
# **Objective:** To analyze the Telco Customer Churn dataset, build a predictive model to identify customers likely to churn, and provide actionable business recommendations.
#
# %% [markdown]
# ## 1. Data Exploration and Cleaning
#
# In this section, we will load the dataset and perform initial exploratory data analysis (EDA). We'll check for missing values, examine data types, and clean the data as needed.
#
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sklearn
# Load the dataset
df = pd.read_csv('Q2 Dataset.csv')

# Initial inspection
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
df.info()

# %% [markdown]
# ### 1.1 Clean the `TotalCharges` Column
#
# The `TotalCharges` column should be numeric, but it's currently an object type. This is likely due to empty spaces for new customers. We will convert it to a numeric type and handle any resulting missing values.

# %%
# Convert 'TotalCharges' to a numeric type. 'coerce' will turn non-numeric values into NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Find out how many missing values were created
print(f"Number of missing values in TotalCharges: {df['TotalCharges'].isnull().sum()}")

# Inspect the rows with missing TotalCharges
# These are typically new customers with 0 tenure.
print("\nRows with missing TotalCharges:")
print(df[df['TotalCharges'].isnull()])

# Fill missing values. Since these are new customers, 0 is a logical fill value.
df['TotalCharges'].fillna(0, inplace=True)

# Verify that the column is now numeric and has no missing values
print("\nDataset Information after cleaning TotalCharges:")
df.info()

# %% [markdown]
# ### 1.2 Exploratory Data Visualization
#
# Now that the data is clean, we'll create some visualizations to better understand the relationships between different features and customer churn.

# %%
# Set the style for the plots
sns.set(style="whitegrid")

# Churn distribution
print("Churn Distribution:")
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()

# %% [markdown]
# #### Numerical Features vs. Churn

# %%
# Plot distribution of numerical features for churned vs. non-churned customers
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
print("\nDistribution of Numerical Features by Churn:")
for feature in numerical_features:
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x=feature, hue='Churn', multiple='stack', kde=True)
    plt.title(f'{feature} Distribution by Churn')
    plt.show()

# %% [markdown]
# #### Categorical Features vs. Churn

# %%
# Plot churn rate for key categorical features
# We'll look at a few of the most impactful ones
categorical_features = ['Contract', 'InternetService', 'OnlineSecurity', 'TechSupport']
print("\nChurn Rate by Key Categorical Features:")
for feature in categorical_features:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=feature, hue='Churn', data=df)
    plt.title(f'Churn Rate by {feature}')
    plt.xticks(rotation=15)
    plt.show()
#
# %% [markdown]
# ## 2. Feature Engineering
#
# Based on the visualizations, we can see that tenure, contract type, and the services a customer has are strong indicators of churn. We will create new features to help the model capture these relationships more effectively.

# %%
# 1. Create a binary feature for new customers (1 year or less)
df['IsNewCustomer'] = (df['tenure'] <= 12).astype(int)

# 2. Create a feature for total number of 'support' services
support_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
df['TotalSupportServices'] = df[support_services].apply(lambda x: x.map({'Yes': 1, 'No': 0, 'No internet service': 0})).sum(axis=1)

# 3. Create a binary feature for customers without any key support
df['NoKeySupport'] = ((df['OnlineSecurity'] == 'No') & (df['TechSupport'] == 'No')).astype(int)


# Display the new features to verify
print("DataFrame with new features (sample):")
print(df[['tenure', 'IsNewCustomer', 'OnlineSecurity', 'TechSupport', 'NoKeySupport', 'TotalSupportServices']].sample(n=10, random_state=42))
#
# %% [markdown]
# ## 3. Data Preprocessing
#
# Now we prepare the data for modeling. This involves:
# 1. Dropping the customerID column.
# 2. Encoding all categorical features into a numeric format.
# 3. Separating the features (X) from the target variable (y).
# 4. Scaling the numerical features.
# 5. Splitting the data into training and testing sets.

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Make a copy to avoid changing the original dataframe during preprocessing
df_model = df.copy()

# Drop customerID as it is not a predictive feature
df_model.drop('customerID', axis=1, inplace=True)

# Convert the target variable 'Churn' to binary
df_model['Churn'] = df_model['Churn'].map({'Yes': 1, 'No': 0})

# Also encode other binary features
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df_model[col] = df_model[col].map({'Yes': 1, 'No': 0})

# Separate features (X) and target (y)
X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

# Identify categorical and numerical features for the preprocessor
# We'll one-hot encode the object columns and scale the numeric ones
categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=np.number).columns

print("Categorical features to be one-hot encoded:")
print(categorical_features)
print("\nNumerical features to be scaled:")
print(numerical_features)

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any)
)

# Split the data into training and testing sets
# We use stratify=y to ensure the churn distribution is the same in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
#
# %% [markdown]
# ## 4. Model Training
#
# Now we will define our models and wrap them in a scikit-learn Pipeline. The pipeline will first apply our preprocessor to the data and then pass the transformed data to the model.
#
# We will train two models to compare:
# 1.  **Logistic Regression**: A simple, interpretable baseline.
# 2.  **Random Forest**: A more powerful ensemble model.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Create the pipelines
# A pipeline chains the preprocessor and the model together.

# Pipeline for Logistic Regression
# We use class_weight='balanced' to help with the imbalanced dataset.
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))
])

# Pipeline for Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Train the models
print("Training Logistic Regression model...")
lr_pipeline.fit(X_train, y_train)
print("Logistic Regression model trained successfully.")

print("\nTraining Random Forest model...")
rf_pipeline.fit(X_train, y_train)
print("Random Forest model trained successfully.")
#
# %% [markdown]
# ## 5. Model Evaluation
#
# Now we'll evaluate our trained models on the test set. We will look at the classification report, confusion matrix, and the ROC-AUC score to determine which model performs better.

# %%
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# --- Evaluate Logistic Regression Model ---
print("--- Logistic Regression Evaluation ---")
y_pred_lr = lr_pipeline.predict(X_test)
print(classification_report(y_test, y_pred_lr, target_names=['No Churn', 'Churn']))

# Plot Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['No Churn', 'Churn'])
disp_lr.plot()
plt.title('Logistic Regression Confusion Matrix')
plt.show()


# --- Evaluate Random Forest Model ---
print("\n--- Random Forest Evaluation ---")
y_pred_rf = rf_pipeline.predict(X_test)
print(classification_report(y_test, y_pred_rf, target_names=['No Churn', 'Churn']))

# Plot Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['No Churn', 'Churn'])
disp_rf.plot()
plt.title('Random Forest Confusion Matrix')
plt.show()


# --- ROC Curve Comparison ---
print("\n--- ROC Curve Comparison ---")
plt.figure(figsize=(8, 6))

# Logistic Regression ROC
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')

# Random Forest ROC
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')

# Plotting
plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
#
# %% [markdown]
# ## 6. Interpretation and Recommendations
#
# Finally, we will interpret the model's results to identify the key drivers of churn and propose actionable recommendations for the business to help retain customers.

# %%
# Extract feature importances from the Random Forest pipeline
feature_importances = rf_pipeline.named_steps['classifier'].feature_importances_

# Get the feature names from the preprocessor
# This includes the one-hot encoded column names
feature_names = rf_pipeline.named_steps['preprocessor'].get_feature_names_out()

# Create a pandas series to view feature importances
importances = pd.Series(feature_importances, index=feature_names)

# Sort features by importance
sorted_importances = importances.sort_values(ascending=False)

# Plot the top 15 features
plt.figure(figsize=(10, 8))
sorted_importances.head(15).sort_values().plot(kind='barh')
plt.title('Top 15 Feature Importances from Random Forest Model')
plt.xlabel('Importance')
plt.show()

# %% [markdown]
# ### Key Questions Answered
#
# #### 1. What are the top indicators of customer churn?
#
# Based on the feature importance plot above, the top indicators of customer churn are:
#
# 1.  **Contract (Month-to-month):** This is consistently the strongest predictor. Customers on rolling monthly contracts are far more likely to leave than those on long-term contracts.
# 2.  **Tenure:** The length of time a customer has been with the company is a major factor. Newer customers are at a much higher risk.
# 3.  **Monthly Charges:** Higher monthly bills are strongly correlated with churn.
# 4.  **Total Charges:** Lower total charges (often linked to shorter tenure) are also a strong indicator.
# 5.  **Internet Service (Fiber Optic):** Customers with Fiber Optic internet service show a higher tendency to churn, which might indicate issues with this specific service (e.g., price, reliability).
# 6.  **Lack of Key Support (`NoKeySupport`):** Our engineered feature proved to be important. Customers without Tech Support or Online Security are more likely to churn.
#
# #### 2. Can a business take proactive steps to retain customers?
#
# Yes. Based on the indicators above, the business can take several data-driven, proactive steps:
#
# *   **Action 1: Target Month-to-Month Customers.**
#     *   **Recommendation:** Create a targeted marketing campaign for customers on month-to-month contracts. Offer them a significant discount (e.g., 15-20%) or a free service upgrade if they switch to a one-year or two-year contract. This directly addresses the #1 churn indicator.
#
# *   **Action 2: Improve Early Customer Experience.**
#     *   **Recommendation:** Since `tenure` is a top factor, implement a robust customer onboarding program for the first 3-6 months. This could include check-in calls, usage tips, and ensuring they are getting the full value of their services. This addresses the high churn rate among new customers.
#
# *   **Action 3: Review Fiber Optic Service/Pricing.**
#     *   **Recommendation:** The high churn rate for fiber optic customers is a red flag. The business should investigate if this is due to service reliability issues, perceived high cost compared to competitors, or poor technical support for this specific service.
#
# *   **Action 4: Upsell Support Services.**
#     *   **Recommendation:** Proactively offer bundled discounts for `TechSupport` and `OnlineSecurity` to customers who don't have them, especially if they also have other risk factors like a month-to-month contract. This can increase customer "stickiness" and reduce churn.


# %%
