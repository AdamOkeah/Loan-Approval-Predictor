import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import seaborn as sns
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv('loan_data.csv')

# Preview Data
print("First five rows of the dataset:")
print(df.head())

# Identify and handle non-numeric columns and apply one-hot encoding
for col in df.columns:
    if df[col].dtype == 'object':  # Check for non-numeric columns
        try:
            # Attempt to convert to datetime and extract features
            df[col] = pd.to_datetime(df[col])
            df[f'{col}_Year'] = df[col].dt.year
            df[f'{col}_Month'] = df[col].dt.month
            df[f'{col}_Day'] = df[col].dt.day
            df = df.drop(columns=[col])  # Drop original date column
        except:
            # For non-date objects, apply one-hot encoding
            df = pd.get_dummies(df, columns=[col], drop_first=True)




# Compute the correlation matrix (only on numeric columns)
correlation_matrix = df.corr()

# Select correlations with the target variable (assuming the target column is 'loan_status')
target_correlation = correlation_matrix['loan_status'].sort_values(ascending=False)

# Display correlations
print("\nCorrelation with Loan Status:")
print(target_correlation)

# Plot correlations with the target variable
plt.figure(figsize=(12, 8))
target_correlation.drop('loan_status').plot(kind='bar')
plt.title('Correlation with Loan Status', fontsize=16)
plt.ylabel('Correlation Coefficient', fontsize=14)
plt.xlabel('Features', fontsize=14)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Select features with correlation above the threshold
threshold = 0.003  # Parameter for deciding which features have a substantial impact on loan status
high_corr_features = correlation_matrix.index[abs(correlation_matrix["loan_status"]) > threshold].tolist()
if "loan_status" in high_corr_features:
    high_corr_features.remove("loan_status")
print("\nHigh Correlated Features:", high_corr_features)

# Define features and target
X = df[high_corr_features]
y = df["loan_status"]



# Scale and normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.astype('float32'))

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print(y_resampled.value_counts())


# Split the resampled data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.4, random_state=42, stratify=y_resampled
)

# Initialize models
knn = KNeighborsClassifier(n_neighbors=4)
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
svm_model = SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale', random_state=42)

# Train models
knn.fit(X_train, y_train)
rf.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Make predictions
knn_pred = knn.predict(X_test)
rf_pred = rf.predict(X_test)
svm_pred = svm_model.predict(X_test)

# Initialize an empty dictionary to store metrics for each model
metrics = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-Score": []
}

# Function to calculate and store metrics
def extract_metrics(model_name, y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics["Model"].append(model_name)
    metrics["Accuracy"].append(round(accuracy_score(y_test, y_pred), 5))
    metrics["Precision"].append(round(report["weighted avg"]["precision"], 5))
    metrics["Recall"].append(round(report["weighted avg"]["recall"], 5))
    metrics["F1-Score"].append(round(report["weighted avg"]["f1-score"], 5))

# Extract metrics for each model
extract_metrics("KNN", y_test, knn_pred)
extract_metrics("Random Forest", y_test, rf_pred)
extract_metrics("SVM", y_test, svm_pred)

# Convert metrics to a DataFrame
metrics_df = pd.DataFrame(metrics)
print("\nMetrics DataFrame:")
print(metrics_df)



# Transpose the metrics DataFrame to switch rows and columns
metrics_df_transposed = metrics_df.set_index("Model").transpose()

# Round the metrics to three decimal points
metrics_df_transposed = metrics_df_transposed.round(5)

print("\nTransposed Metrics DataFrame:")
print(metrics_df_transposed)

# Plot the table using matplotlib
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(
    cellText=metrics_df_transposed.values,
    rowLabels=metrics_df_transposed.index,
    colLabels=metrics_df_transposed.columns,
    cellLoc='center',
    loc='center'
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.auto_set_column_width(col=list(range(len(metrics_df_transposed.columns))))

# Add a title and display the table
plt.title("Model Metrics Comparison (Rounded to 3 Decimal Points)", fontsize=16, fontweight='bold')
plt.show()

