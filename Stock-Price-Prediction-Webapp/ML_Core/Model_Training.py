import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

url = "https://www.dropbox.com/scl/fi/qcoe6o50xkzcdx284afjw/yahoo_stock.csv?rlkey=i5zy458vi042zc2eegkuefty3&st=ljs8pgur&dl=1"
df = pd.read_csv(url)

# Preprocessing alignment
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date').reset_index(drop=True)

# Calculating the indicators requested by your task
df['SMA_5'] = df['Adj Close'].rolling(window=5).mean()
df['SMA_20'] = df['Adj Close'].rolling(window=20).mean()
df['EMA_14'] = df['Adj Close'].ewm(span=14, adjust=False).mean()

# RSI Calculation
delta = df['Adj Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / (loss + 1e-10) 
df['RSI'] = 100 - (100 / (1 + rs))

df['Daily_Return'] = df['Adj Close'].pct_change()
df['Prev_Close'] = df['Adj Close'].shift(1)
df['Prev_Volume'] = df['Volume'].shift(1)

# Target configuration (Predicting next day's price)
df['Target'] = df['Adj Close'].shift(-1)
df_clean = df.dropna().copy()

# Feature mapping
features = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'SMA_5', 'SMA_20', 'EMA_14', 'RSI', 'Daily_Return', 'Prev_Close', 'Prev_Volume']
X = df_clean[features]
y = df_clean['Target']

# Spliting the dataset and Training the Model
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Fit Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate
preds = model.predict(X_test_scaled)
print(f"R2 Score: {r2_score(y_test, preds):.4f}\nMAE: {mean_absolute_error(y_test, preds):.2f}")

current_dir = os.path.dirname(__file__)
joblib.dump(model, os.path.join(current_dir, 'stock_model.pkl'))
joblib.dump(scaler, os.path.join(current_dir, 'scaler.pkl'))

print("Model Trained and Saved")