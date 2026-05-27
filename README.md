# InsightSol Technologies AI/ML Internship

Welcome to my official portfolio repository showcasing the machine learning applications developed during my remote AI Engineering Internship at **InsightSol Technologies**. This repository contains end-to-end predictive analytics web applications, demonstrating robust data workflows, feature engineering, model optimization, and clean deployment interfaces.

---

## 📌 Projects Overview & Scope

During this internship, I engineered two primary production-ready predictive applications designed to solve real-world valuation and forecasting tasks.

### 🚗 1. Used Car Price Prediction Webapp
This project is a comprehensive data science solution that calculates the market resale valuation of a vehicle. Instead of making rough guesses, it applies a trained machine learning pipeline to analyze complex, non-linear feature interactions to provide precise estimations.
* **Core Machine Learning Engine:** Powered by a **Random Forest Regressor** pipeline, carefully optimized to minimize mean absolute errors and prevent overfitting.
* **Data Features Handled:** The model processes critical structural and historical metrics, including:
  * **Vehicle Age:** Dynamically derived from the manufacturing year.
  * **Usage Wear & Tear:** Quantified via total kilometers driven.
  * **Mechanical Attributes:** Engine displacement (cc) and power metrics.
  * **Categorical Variables:** Automatically encodes fuel types (Petrol/Diesel/CNG) and transmission systems (Manual/Automatic).
* **Web UI Capabilities:** Built with a responsive **Flask** backend framework that handles dynamic web form client requests, sanitizes the user data, feeds it directly into the serialized pipeline, and renders the calculated price instantly on screen.

### 📈 2. Stock Price Prediction Webapp
This project is a financial analytics tool deployed to evaluate asset pricing trends and calculate performance projections. It focuses on handling continuous time-series style data sequences to estimate market momentum.
* **Core Machine Learning Engine:** Utilizes predictive **Regression Analysis architectures** tailored for continuous numerical tracking patterns.
* **Data Features Handled:** Parses underlying fundamental quantitative market indicators, including:
  * **Price Thresholds:** Historical opening and closing daily values.
  * **Market Volatility:** High/low intraday spreads.
  * **Trading Volume:** The aggregate volume of shares moving through transactions.
  * **Moving Averages:** Trend-smoothing variables to measure stock momentum.
* **Web UI Capabilities:** Features a clean, data-focused interactive dashboard layout. It includes data integrity boundaries to ensure seamless exception handling for volatile, extreme, or zero-bound numerical inputs, preventing application layer crashes.

---

## 📂 Repository Architecture
```plaintext
InSightSol-Technologies-Internship/
│
├── Car-Price-Prediction-Webapp/      # Vehicle resale valuation app
│   ├── app.py                         # Main Flask application layer
│   ├── car_model.pkl                  # Trained Random Forest pipeline (Git LFS)
│   └── templates/                     # Frontend UI assets (HTML/CSS layout)
│
└── Stock-Price-Prediction-Webapp/    # Financial trend forecasting app
    ├── app.py                         # Main application runtime layer
    ├── stock_model.pkl                # Serialized predictive model (Git LFS)
    └── templates/                     # Web forms and layout interface
