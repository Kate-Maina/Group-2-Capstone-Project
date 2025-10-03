from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the model
model = joblib.load('random_forest.pkl')

# Configuration - must match training
LAGS = [1, 2, 3, 4, 8, 12, 26, 52]
ROLLS = [4, 8, 12, 26]
DIFFS = [1, 4]

# Shop code mapping (must match training order)
SHOP_MAP = {
    "901 - Gikomba": 0,
    "904 - Kericho": 1,
    "904.5 Meru WS": 2,
    "905 - Eldoret": 3,
    "906 - Kisii": 4,
    "907-Kisumu WS": 5,
    "910 - Kitale": 6,
    "913 -Kakamega WS": 7,
    "914-EMBU WS": 8,
    "915-NANYUKI WS": 9,
    "99 Warehouse": 10
}

def calculate_features(sales_history, prediction_date, shop_code):
    """
    Calculate all required features from sales history
    
    Args:
        sales_history: list/array of recent weekly sales (at least 52 weeks)
        prediction_date: datetime object for prediction
        shop_code: integer shop code
    
    Returns:
        dict of features
    """
    history = np.array(sales_history, dtype=float)
    
    features = {}
    
    # Lag features
    for k in LAGS:
        features[f"lag_{k}"] = history[-k] if len(history) >= k else 0.0
    
    # Rolling means (on past values only)
    for w in ROLLS:
        if len(history) >= 1:
            features[f"roll{w}"] = np.mean(history[-min(w, len(history)):])
        else:
            features[f"roll{w}"] = 0.0
    
    # Diffs: y(t-1) - y(t-1-d)
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
    
    # Shop code
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
    """
    Predict weekly sales for a shop
    
    Expected JSON format:
    {
        "shop": "901 - Gikomba",
        "date": "2025-10-06",  # Monday of the week to predict (YYYY-MM-DD)
        "sales_history": [1500, 1600, 1450, ..., 1700]  # At least 52 weeks of history
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'success': False
            }), 400
        
        required_fields = ['shop', 'date', 'sales_history']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                'error': f'Missing required fields: {missing}',
                'success': False
            }), 400
        
        shop_name = data['shop']
        date_str = data['date']
        sales_history = data['sales_history']
        
        # Validate shop
        if shop_name not in SHOP_MAP:
            return jsonify({
                'error': f'Unknown shop: {shop_name}',
                'available_shops': list(SHOP_MAP.keys()),
                'success': False
            }), 400
        
        shop_code = SHOP_MAP[shop_name]
        
        # Validate date
        try:
            pred_date = pd.to_datetime(date_str)
        except:
            return jsonify({
                'error': 'Invalid date format. Use YYYY-MM-DD',
                'success': False
            }), 400
        
        # Validate sales history
        if not isinstance(sales_history, list) or len(sales_history) < 52:
            return jsonify({
                'error': 'sales_history must be a list with at least 52 weeks of data',
                'received_length': len(sales_history) if isinstance(sales_history, list) else 0,
                'success': False
            }), 400
        
        # Calculate features
        features = calculate_features(sales_history, pred_date, shop_code)
        
        # Create feature vector in correct order
        feature_cols = (
            [f"lag_{k}" for k in LAGS] +
            [f"roll{w}" for w in ROLLS] +
            [f"diff{d}" for d in DIFFS] +
            ["dow", "week", "month", "quarter", "year", "shop_code"]
        )
        
        X = np.array([[features[col] for col in feature_cols]], dtype=float)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # If model was trained with log transform, reverse it
        # Based on your code: USE_LOG_FE = True
        prediction = np.expm1(prediction)  # reverse log1p
        
        return jsonify({
            'shop': shop_name,
            'prediction_date': date_str,
            'predicted_sales': float(prediction),
            'features_used': features,
            'success': True
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    