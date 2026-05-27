from flask import Flask, render_template, request, jsonify
import os
import joblib
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'ML_Core', 'stock_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'ML_Core', 'scaler.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Backend System Alert: ML Model and Preprocessing Scaler loaded successfully!")
except Exception as e:
    print(f"Backend System Error: Failed to load pkl files. Details: {e}")

@app.route('/')
def home():

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        input_features = [
            float(request.form['Open']),
            float(request.form['High']),
            float(request.form['Low']),
            float(request.form['Adj_Close']),
            float(request.form['Volume']),
            float(request.form['SMA_5']),
            float(request.form['SMA_20']),
            float(request.form['EMA_14']),
            float(request.form['RSI']),
            float(request.form['Daily_Return']),
            float(request.form['Prev_Close']),
            float(request.form['Prev_Volume'])
        ]
        

        features_array = np.array([input_features])
        
        scaled_features = scaler.transform(features_array)
        
        prediction_output = model.predict(scaled_features)[0]
        
        return render_template('index.html', 
                               prediction_text=f'Forecasted Next-Day Closing Price: ${prediction_output:.2f}')
        
    except Exception as e:
        return render_template('index.html', prediction_text=f"Execution Error: Please ensure all metrics are filled properly.")

if __name__ == '__main__':
    app.run(debug=True)
