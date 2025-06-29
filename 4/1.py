import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Load local dataset (ensure student-mat.csv is in your directory)
df = pd.read_csv("student-mat.csv", sep=';')

# 2. Create performance class based on G3
df['performance'] = df['G3'].apply(lambda x: 'poor' if x < 10 else 'average' if x < 13 else 'good' if x < 16 else 'outstanding')

# 3. Encode categorical variables
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# 4. Feature Selection (based on SSLF)
features = ['failures', 'G1', 'absences', 'internet']
X = df[features]
y = df['performance']

# 5. First-tier: PASS/FAIL classification
y_pass_fail = df['G3'].apply(lambda x: 'pass' if x >= 10 else 'fail')
X_train, X_test, y_train, y_test = train_test_split(X, y_pass_fail, test_size=0.3, random_state=42)

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_first = nb.predict(X_test)

print("=== First-Tier (PASS/FAIL) ===")
print(classification_report(y_test, y_pred_first))

# 6. Filter 'pass' students for second-tier classification
pass_indices = y_test[y_test == 'pass'].index
X_pass = X_test.loc[pass_indices]
y_second = y.loc[pass_indices]

# 7. Second-tier: Multi-class classification using stacking
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('ada', AdaBoostClassifier(n_estimators=50, random_state=42))
]
stack = StackingClassifier(estimators=base_learners, final_estimator=RandomForestClassifier())
stack.fit(X_pass, y_second)
y_pred_second = stack.predict(X_pass)

print("\n=== Second-Tier (Outstanding, Good, Average) ===")
print(classification_report(y_second, y_pred_second))
print(f"Accuracy: {accuracy_score(y_second, y_pred_second) * 100:.2f}%")

# 8. Graphs and Visualizations

# A. Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_second, y_pred_second), annot=True, fmt='d', cmap='Blues',
            xticklabels=stack.classes_, yticklabels=stack.classes_)
plt.title('Confusion Matrix - Second Tier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# B. Distribution of classified performance
plt.figure(figsize=(6, 4))
sns.countplot(x=y_pred_second, order=['poor', 'average', 'good', 'outstanding'])
plt.title('Student Performance Classification Distribution')
plt.xlabel('Performance Category')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()

# C. Accuracy bar chart
performance_scores = accuracy_score(y_second, y_pred_second) * 100
plt.figure(figsize=(4, 3))
plt.bar(['MTSPEM Model'], [performance_scores], color='green')
plt.ylabel('Accuracy (%)')
plt.title('MTSPEM Model Accuracy')
plt.ylim(0, 100)
plt.tight_layout()
plt.show()
