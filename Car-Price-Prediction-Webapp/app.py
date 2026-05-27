from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the upgraded structural constraint model file
with open('car_predictor.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
encoders = bundle['encoders']

@app.route('/')
def home():
    return render_template('index.html', current_brand=None, current_model=None, current_variant=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Capture form fields from frontend selection templates safely
            brand_code = int(request.form['brand'])
            model_code = int(request.form['model'])
            variant_code = int(request.form['variant'])
            year = int(request.form['year'])
            km_driven = float(request.form['km_driven'])
            fuel_code = int(request.form['fuel_type'])
            transmission_code = int(request.form['transmission'])
            seller_code = int(request.form['seller_type'])
            owner_code = int(request.form['owner'])
            
            # 2. Build structured DataFrame array matching model layout EXACTLY (9 Columns)
            input_data = [[year, km_driven, fuel_code, seller_code, transmission_code, owner_code, brand_code, model_code, variant_code]]
            feature_names = ['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'brand', 'model', 'variant']
            test_df = pd.DataFrame(input_data, columns=feature_names)
            
            # 3. Core evaluation & PKR Conversion Pipeline (1 INR = 2.91 PKR)
            prediction_inr = model.predict(test_df)[0]
            EXCHANGE_RATE = 2.91
            prediction_pkr = prediction_inr * EXCHANGE_RATE
            
            # 4. Format currency representation into PKR standard Lakh layout
            if prediction_pkr >= 100000:
                lakh_value = prediction_pkr / 100000
                formatted_price = f"Rs. {lakh_value:.2f} Lakh"
            else:
                formatted_price = f"Rs. {prediction_pkr:,.2f}"
            
            # 5. Return execution states back to refresh elements dynamically
            return render_template('index.html', 
                                   prediction_text=formatted_price, 
                                   raw_price=prediction_pkr,
                                   current_brand=brand_code,
                                   current_model=model_code,
                                   current_variant=variant_code,
                                   current_year=year,
                                   current_km=int(km_driven),
                                   current_fuel=fuel_code,
                                   current_trans=transmission_code,
                                   current_seller=seller_code,
                                   current_owner=owner_code)
            
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error processing telemetry variables: {str(e)}')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)