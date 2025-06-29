# Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Load Dataset
data = load_wine()
X, y = data.data, data.target
feature_names = data.feature_names
class_names = data.target_names

# Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Store results
results = {}

# 1. Decision Tree
dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
results['Decision Tree'] = accuracy_score(y_test, y_pred_dt)

# 2. Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
results['Naive Bayes'] = accuracy_score(y_test, y_pred_nb)

# 3. SVM
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
results['SVM'] = accuracy_score(y_test, y_pred_svm)

# 4. ANN
ann = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
ann.fit(X_train, y_train)
y_pred_ann = ann.predict(X_test)
results['ANN'] = accuracy_score(y_test, y_pred_ann)

# -----------------------------
# 🔍 Classification Reports
# -----------------------------
print("=== Classification Reports ===\n")
print("Decision Tree:\n", classification_report(y_test, y_pred_dt, target_names=class_names))
print("Naive Bayes:\n", classification_report(y_test, y_pred_nb, target_names=class_names))
print("SVM:\n", classification_report(y_test, y_pred_svm, target_names=class_names))
print("ANN:\n", classification_report(y_test, y_pred_ann, target_names=class_names))

# -----------------------------
# 📊 1. Accuracy Comparison Bar Chart
# -----------------------------
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color='teal')
plt.ylabel('Accuracy')
plt.title('ML Algorithm Accuracy Comparison')
plt.ylim(0.85, 1.0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# -----------------------------
# 📈 2. Confusion Matrix (for Decision Tree as example)
# -----------------------------
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -----------------------------
# 🌲 3. Decision Tree Plot
# -----------------------------
plt.figure(figsize=(20, 10))
plot_tree(dt, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
plt.title("ID3 Decision Tree Visualization")
plt.tight_layout()
plt.show()
