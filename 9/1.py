# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load Sample Dataset (you can replace this with your actual CSV)
# Example structure: ['Age', 'Gender', 'Course_Type', 'Activity_Score', 'Completion_Status']
data = pd.read_csv("mooc_dataset.csv")  # Replace with your dataset path

# Data Preprocessing
data.dropna(inplace=True)  # Remove missing values

# Encoding categorical variables
data_encoded = pd.get_dummies(data, drop_first=True)

# Feature and Target split
X = data_encoded.drop("Completion_Status", axis=1)  # Target: 0 = Dropout, 1 = Completed
y = data_encoded["Completion_Status"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training - Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance Plot
feat_importance = pd.Series(rf.feature_importances_, index=X.columns)
feat_importance.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()
