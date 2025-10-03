from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the bundle (model was saved as a dictionary)
bundle = joblib.load('random_forest.pkl')
model = bundle['model']
feature_cols = bundle['feature_cols']
shop_code_map = bundle['shop_code_map']

# Configuration - must match training
LAGS = [1, 2, 3, 4, 8, 12, 26, 52]
ROLLS = [4, 8, 12, 26]
DIFFS = [1, 4]

# Get shop names from the loaded shop_code_map
SHOP_MAP = shop_code_map

def calculate_features(sales_history, prediction_date, shop_code):
    """
    Calculate all required features from sales history
    """
    history = np.array(sales_history, dtype=float)
    
    features = {}
    
    # Lag features
    for k in LAGS:
        features[f"lag_{k}"] = history[-k] if len(history) >= k else 0.0
    
    # Rolling means
    for w in ROLLS:
        if len(history) >= 1:
            features[f"roll{w}"] = np.mean(history[-min(w, len(history)):])
        else:
            features[f"roll{w}"] = 0.0
    
    # Diffs
    for d in DIFFS:
        if len(history) >= (1 + d):
            features[f"diff{d}"] = history[-1] - history[-1-d]
        else:
            features[f"diff{d}"] = 0.0
    
    # Calendar features
    features["dow"] = prediction_date.weekday()
    features["week"] = prediction_date.isocalendar()[1]
    features["month"] = prediction_date.month
    features["quarter"] = (prediction_date.month - 1) // 3 + 1
    features["year"] = prediction_date.year
    features["shop_code"] = shop_code
    
    return features


@app.route('/')
def home():
    return jsonify({
        "message": "Sales Prediction API is running!",
        "endpoints": {
            "/predict": "POST - Predict weekly sales",
            "/shops": "GET - List available shops",
            "/health": "GET - Health check"
        }
    })


@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": True})


@app.route('/shops')
def shops():
    return jsonify({
        "shops": list(SHOP_MAP.keys()),
        "note": "Use exact shop names in prediction requests"
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided', 'success': False}), 400
        
        required_fields = ['shop', 'date', 'sales_history']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({'error': f'Missing required fields: {missing}', 'success': False}), 400
        
        shop_name = data['shop']
        date_str = data['date']
        sales_history = data['sales_history']
        
        if shop_name not in SHOP_MAP:
            return jsonify({
                'error': f'Unknown shop: {shop_name}',
                'available_shops': list(SHOP_MAP.keys()),
                'success': False
            }), 400
        
        shop_code = SHOP_MAP[shop_name]
        
        try:
            pred_date = pd.to_datetime(date_str)
        except:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD', 'success': False}), 400
        
        if not isinstance(sales_history, list) or len(sales_history) < 52:
            return jsonify({
                'error': 'sales_history must be a list with at least 52 weeks of data',
                'received_length': len(sales_history) if isinstance(sales_history, list) else 0,
                'success': False
            }), 400
        
        features = calculate_features(sales_history, pred_date, shop_code)
        X = np.array([[features[col] for col in feature_cols]], dtype=float)
        prediction = model.predict(X)[0]
        prediction = np.expm1(prediction)
        
        return jsonify({
            'shop': shop_name,
            'prediction_date': date_str,
            'predicted_sales': float(prediction),
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)