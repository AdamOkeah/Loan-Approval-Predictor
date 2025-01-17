import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv('loan_data.csv')

# Identify and handle non-numeric columns
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

# Check the structure of the data after handling categorical columns
print(f"Shape after preprocessing: {df.shape}")

# Compute the correlation matrix (only on numeric columns)
correlation_matrix = df.corr()

# Select correlations with the target variable (assuming the target column is 'Loan_Status')
target_correlation = correlation_matrix['loan_status'].sort_values(ascending=False)

# Display correlations
print("Correlation with Loan Status:")
print(target_correlation)

# Plot correlations with the target variable
plt.figure(figsize=(10, 6))
target_correlation.drop('loan_status').plot(kind='bar')
plt.title('Correlation with Loan Status')
plt.ylabel('Correlation Coefficient')
plt.xlabel('Features')
plt.show()

threshold = 0.001
correlation_matrix = df.corr()
high_corr_features = correlation_matrix.index[abs(correlation_matrix["loan_status"]) > threshold].tolist()
high_corr_features.remove("loan_status")
print(high_corr_features)
X = df[high_corr_features]
y = df["loan_status"]
print(y.value_counts())

# Scale and normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X.astype('float32'))

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Proceed with train-test split
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)

# Define the DNN model
model = Sequential([
    Dense(32, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=l2(0.02)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=128,
    callbacks=[early_stopping]
)

# Evaluate the model on the test set
evaluation = model.evaluate(X_test, y_test)
print(f"Test Loss: {evaluation[0]}")
print(f"Test Accuracy: {evaluation[1]}")

plt.figure(figsize=(14, 6))

# Training and Validation Loss Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yticks(np.arange(0.25, 0.75, 0.02))  # Custom y-ticks with step size of 0.02

# Training and Validation Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.yticks(np.arange(0.5, 0.9, 0.05))  # Custom y-ticks with step size of 0.005

plt.tight_layout()
plt.show()

conf_matrix = confusion_matrix(y_test, dnn_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("DNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()