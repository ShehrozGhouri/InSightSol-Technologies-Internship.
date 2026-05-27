import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import HistGradientBoostingRegressor # Switched to Constraint-Locked Boosting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# 1. Load the dataset
df = pd.read_csv('https://www.dropbox.com/scl/fi/p7nbfatqu7ygrqess0oq0/Cars-detail-for-model.csv?rlkey=l8amm4dbgspt1hq3coom0nben&st=vgis5hhl&dl=1')

# 2. Extract features: Brand, Base Model, and Variant
def parse_car_name(name):
    words = name.split()
    brand = words[0]
    model = words[1] if len(words) > 1 else 'unknown'
    variant = " ".join(words[2:]) if len(words) > 2 else 'Standard'
    return pd.Series([brand, model, variant])

df[['brand', 'model', 'variant']] = df['name'].apply(parse_car_name)

# 3. Fit categorical fields into LabelEncoders
le_brand = LabelEncoder()
le_model = LabelEncoder()
le_variant = LabelEncoder()
le_fuel = LabelEncoder()
le_seller = LabelEncoder()
le_trans = LabelEncoder()
le_owner = LabelEncoder()

df['brand_enc'] = le_brand.fit_transform(df['brand'])
df['model_enc'] = le_model.fit_transform(df['model'])
df['variant_enc'] = le_variant.fit_transform(df['variant'])
df['fuel_enc'] = le_fuel.fit_transform(df['fuel'])
df['seller_enc'] = le_seller.fit_transform(df['seller_type'])
df['trans_enc'] = le_trans.fit_transform(df['transmission'])
df['owner_enc'] = le_owner.fit_transform(df['owner'])

# 4. Construct final structured feature matrix (9 Features)
X = pd.DataFrame({
    'year': df['year'],
    'km_driven': df['km_driven'],
    'fuel': df['fuel_enc'],
    'seller_type': df['seller_enc'],
    'transmission': df['trans_enc'],
    'owner': df['owner_enc'],
    'brand': df['brand_enc'],
    'model': df['model_enc'],
    'variant': df['variant_enc']
})
y = df['selling_price']

# 5. Split data for verification check (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# 6. Inject Monotonic Constraints Mapping
# 1  = Direct relationship (Higher year = Higher price)
# -1 = Inverse relationship (Higher mileage = Lower price)
# 0  = No forced rules for categorical flags
rules_matrix = [1, -1, 0, 0, 0, 0, 0, 0, 0]

model = HistGradientBoostingRegressor(monotonic_cst=rules_matrix, random_state=2)
model.fit(X_train, y_train)
score = r2_score(y_test, model.predict(X_test))
print(f"R-squared Score: {score:.4f}")

# 7. Fit Final Model on 100% of data parameters for deployment 
production_model = HistGradientBoostingRegressor(monotonic_cst=rules_matrix, random_state=2)
production_model.fit(X, y)

# 8. Package everything into a deployment bundle dictionary
bundle = {
    'model': production_model,
    'encoders': {
        'brand': le_brand,
        'model': le_model,
        'variant': le_variant,
        'fuel': le_fuel,
        'seller_type': le_seller,
        'transmission': le_trans,
        'owner': le_owner
    }
}

# 9. Save directly to a single bundle binary pkl file
with open('car_predictor.pkl', 'wb') as f:
    pickle.dump(bundle, f)

print("model successfully saved to 'car_predictor.pkl'!")
