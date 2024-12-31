import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# Load the dataset
data = pd.read_csv('medical_data.csv')

# Check if the dataset is loaded correctly
print("Dataset Preview:")
print(data.head())

# Define features (X) and target (y)
X = data[['age', 'symptom_fever', 'symptom_cough', 'symptom_fatigue', 'medical_history']]  # Features
y = data['diagnosis']  # Target (diagnosis)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display Confusion Matrix as a Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y.unique(), yticklabels=y.unique())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report (with zero_division=1 to avoid warnings)
print("\nClassification Report:")
report = classification_report(y_test, y_pred, zero_division=1)
print(report)

# To visualize the Classification Report as a heatmap
report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
report_df = pd.DataFrame(report_dict).transpose()

# Plot Classification Report as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Classification Report Heatmap')
plt.show()