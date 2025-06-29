# Required Libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import CCIIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Step 1: Download Historical Data (NIFTY Index or TCS stock)
ticker = "^NSEI"  # Use "TCS.NS" for TCS
data = yf.download(ticker, start="2020-01-01", end="2024-12-31", auto_adjust=False)

# Step 2: Clean the Data
data.dropna(inplace=True)
data.columns = data.columns.get_level_values(0)  # In case columns are multi-level

# Step 3: Generate Technical Indicators
data['RSI'] = RSIIndicator(close=data['Close']).rsi()
data['Momentum'] = ROCIndicator(close=data['Close']).roc()
data['CCI'] = CCIIndicator(high=data['High'], low=data['Low'], close=data['Close']).cci()

# Drop rows with NaN due to indicators
data.dropna(inplace=True)

# Step 4: Generate Target - 1 if price goes up next day, else 0
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

# Step 5: Prepare Feature Matrix and Label Vector
features = ['RSI', 'Momentum', 'CCI']
X = data[features]
y = data['Target']

# Step 6: Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Step 8: Define and Train Models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF Kernel)': SVC(kernel='rbf', C=1.0, gamma=0.5, probability=True),
    'Artificial Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, learning_rate_init=0.01, random_state=42)
}

results = {}

# Step 9: Train, Evaluate, and Store Results
for name, model in models.items():
    print(f"\nTraining and Evaluating: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    results[name] = {'Accuracy': acc, 'F1-Score': f1}

# Step 10: Plot Results
results_df = pd.DataFrame(results).T
plt.figure(figsize=(10, 6))
results_df.plot(kind='bar', colormap='Set2', edgecolor='black')
plt.title('Model Performance on Predicting Next Day Price Direction')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
