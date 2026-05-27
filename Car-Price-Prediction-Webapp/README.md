# 🚗 Used Car Price Prediction Webapp

This sub-project consists of a data-driven web application that predicts the estimated resale value of a used vehicle based on multiple performance, wear, and structural features. It bridges a complex ensemble learning model with an intuitive user interface.

## 🧠 Machine Learning Engine & Logic
* **Algorithm Selection:** Built using a **Random Forest Regressor** pipeline. This ensemble method aggregates multiple decision trees to minimize mean absolute variance, handling complex non-linear columns smoothly while preventing overfitting.
* **Feature Engineering Pipeline:**
  * **Vehicle Age:** Dynamically derived during preprocessing based on the manufacturing year.
  * **Wear Parameter:** Continuous monitoring of total kilometers driven.
  * **Mechanical Attributes:** Evaluates engine displacement size (cc) and output power variables.
  * **Categorical Feature Encoding:** Seamless handling and pipeline transformation for non-numeric fields like Fuel Classification (Petrol/Diesel/CNG) and Transmission Mechanics (Manual/Automatic).

## 🛠️ Web Interface & Deployment
* **Framework:** Powered by a clean **Flask** web infrastructure.
* **Architecture:** Form fields process HTTP POST requests, capture inputs, pass arrays safely through the serialized model wrapper (`car_predict.pkl`), and render the predicted output back to the client interface dynamically without app-layer delays.

## 🏃‍♂️ How to Run This Webapp Separately
If you want to spin up just this specific application, execute these commands from the root folder:
```bash
cd Car-Price-Prediction-Webapp
python app.py
