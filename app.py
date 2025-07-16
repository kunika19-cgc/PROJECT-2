from flask import Flask, render_template, request
import pickle
import numpy as np
import logging
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and scaler with error handling
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model/scaler: {str(e)}")
    raise RuntimeError("Failed to load model/scaler") from e

# Expected feature names and validation ranges
FEATURE_INFO = {
    'age': (0, 120),
    'sex': (0, 1),
    'cp': (0, 3),
    'trestbps': (0, 300),  # resting blood pressure
    'chol': (0, 600),      # serum cholesterol
    'fbs': (0, 1),         # fasting blood sugar
    'restecg': (0, 2),
    'thalach': (0, 250),   # maximum heart rate
    'exang': (0, 1),       # exercise induced angina
    'oldpeak': (0, 10),    # ST depression
    'slope': (0, 2),
    'ca': (0, 3),          # number of major vessels
    'thal': (0, 3)
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate input data
        if not request.form:
            raise BadRequest("No form data submitted")
        
        # Collect and validate features
        features = []
        for feature_name in FEATURE_INFO:
            try:
                value = float(request.form.get(feature_name, ''))
                min_val, max_val = FEATURE_INFO[feature_name]
                if not min_val <= value <= max_val:
                    raise ValueError(f"{feature_name} must be between {min_val} and {max_val}")
                features.append(value)
            except ValueError as e:
                raise BadRequest(f"Invalid value for {feature_name}: {str(e)}")
        
        # Verify correct number of features
        if len(features) != len(FEATURE_INFO):
            raise BadRequest(f"Expected {len(FEATURE_INFO)} features, got {len(features)}")
        
        # Scale features and make prediction
        final_features = scaler.transform([features])
        prediction = model.predict(final_features)
        proba = model.predict_proba(final_features)[0]
        
        # Prepare results
        result = {
            'prediction': int(prediction[0]),
            'probability': float(proba[prediction[0]]),
            'class_probs': {
                'No Heart Disease': float(proba[0]),
                'Heart Disease': float(proba[1])
            },
            'features': dict(zip(FEATURE_INFO.keys(), features))
        }
        
        # Format output
        if result['prediction'] == 1:
            output = f"Heart Disease Detected (Confidence: {result['probability']*100:.1f}%)"
        else:
            output = f"No Heart Disease Detected (Confidence: {result['probability']*100:.1f}%)"
            
        return render_template('index.html', 
                             prediction_text=output,
                             result_details=result)
    
    except BadRequest as e:
        logger.warning(f"Bad request: {str(e)}")
        return render_template('index.html', 
                            prediction_text=f"Input Error: {str(e)}",
                            error=True), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return render_template('index.html', 
                            prediction_text="An unexpected error occurred",
                            error=True), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)