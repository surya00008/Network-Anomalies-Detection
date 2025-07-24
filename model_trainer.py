import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib # For saving/loading models
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
preprocessed_data_path = 'preprocessed_data.csv' # Input: CSV from preprocessing.ipynb
model_save_path = 'isolation_forest_model.joblib' # Output: Trained model
threshold_save_path = 'anomaly_threshold.txt' # Output: Determined threshold

# --- Step 2: Load Preprocessed Data ---
try:
    df_preprocessed = pd.read_csv(preprocessed_data_path)
    print(f"Successfully loaded {len(df_preprocessed)} rows from {preprocessed_data_path}")
    print("\nPreprocessed DataFrame Head:")
    print(df_preprocessed.head())
    print("\nPreprocessed DataFrame Info:")
    df_preprocessed.info()
except FileNotFoundError:
    print(f"Error: The file '{preprocessed_data_path}' was not found.")
    print("Please ensure preprocessing.ipynb ran successfully and the CSV exists.")
    exit() # Terminate script if input data is missing

# --- Step 4: Model Training ---
# Extract the features (all columns in your preprocessed DataFrame are now features)
X = df_preprocessed.values # Convert DataFrame to NumPy array for model training

# Initialize the Isolation Forest model
# 'contamination' is a hyperparameter: it's the proportion of outliers in the dataset.
# Adjust this value based on your expected anomaly rate.
model = IsolationForest(contamination=0.01, random_state=42) # Using 1% contamination as a starting point

print(f"\nTraining Isolation Forest model with contamination={model.contamination}...")
model.fit(X) # Train the model on your preprocessed data
print("Model training complete.")

# Get anomaly scores and predictions
df_preprocessed['anomaly_score'] = model.decision_function(X)
df_preprocessed['is_anomaly_predicted'] = model.predict(X)

print("\nFirst 10 rows with Anomaly Score and Prediction:")
print(df_preprocessed[['anomaly_score', 'is_anomaly_predicted']].head(10))

num_detected_anomalies = df_preprocessed[df_preprocessed['is_anomaly_predicted'] == -1].shape[0]
print(f"\nNumber of detected anomalies (based on model's contamination={model.contamination}): {num_detected_anomalies}")
print(f"Proportion of detected anomalies: {num_detected_anomalies / len(df_preprocessed):.4f}")


# --- Step 5: Thresholding for Anomaly Detection ---
plt.figure(figsize=(12, 6))
sns.histplot(df_preprocessed['anomaly_score'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Anomaly Scores from Isolation Forest', fontsize=16)
plt.xlabel('Anomaly Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

min_score = df_preprocessed['anomaly_score'].min()
mean_score = df_preprocessed['anomaly_score'].mean()
contamination_threshold = df_preprocessed[df_preprocessed['is_anomaly_predicted'] == -1]['anomaly_score'].max() if num_detected_anomalies > 0 else df_preprocessed['anomaly_score'].min()

plt.axvline(x=min_score, color='red', linestyle='--', label=f'Min Score: {min_score:.2f}')
plt.axvline(x=mean_score, color='blue', linestyle='--', label=f'Mean Score: {mean_score:.2f}')
if num_detected_anomalies > 0:
    plt.axvline(x=contamination_threshold, color='green', linestyle=':', label=f'Contamination Threshold: {contamination_threshold:.2f}')
plt.legend(fontsize=10)
plt.show()

plt.figure(figsize=(10, 2))
sns.boxplot(x=df_preprocessed['anomaly_score'], color='lightcoral')
plt.title('Box Plot of Anomaly Scores', fontsize=14)
plt.xlabel('Anomaly Score', fontsize=12)
plt.show()

print(f"\nMinimum Anomaly Score: {min_score:.4f}")
print(f"Maximum Anomaly Score: {df_preprocessed['anomaly_score'].max():.4f}")
print(f"Average Anomaly Score: {mean_score:.4f}")

# Choose a threshold based on a percentile (e.g., 5th percentile)
chosen_threshold = np.percentile(df_preprocessed['anomaly_score'], 5)

print(f"\nChosen Anomaly Threshold (5th percentile): {chosen_threshold:.4f}")

df_preprocessed['custom_is_anomaly'] = (df_preprocessed['anomaly_score'] <= chosen_threshold).astype(int)
df_preprocessed['custom_is_anomaly'] = df_preprocessed['custom_is_anomaly'].replace({0: 1, 1: -1})

num_custom_anomalies = df_preprocessed[df_preprocessed['custom_is_anomaly'] == -1].shape[0]
print(f"Number of anomalies detected with custom threshold: {num_custom_anomalies}")
print(f"Proportion of anomalies detected with custom threshold: {num_custom_anomalies / len(df_preprocessed):.4f}")


# --- Step 6: Save the Trained Model and Threshold ---
joblib.dump(model, model_save_path)
print(f"\nTrained Isolation Forest model saved to: {model_save_path}")

with open(threshold_save_path, 'w') as f:
    f.write(str(chosen_threshold))
print(f"Anomaly detection threshold ({chosen_threshold:.4f}) saved to: {threshold_save_path}")