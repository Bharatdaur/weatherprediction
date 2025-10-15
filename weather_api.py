from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

app = Flask(__name__)
CORS(app)

class WeatherAPI:
    def __init__(self):
        self.condition_model = None
        self.temp_model = None
        self.humidity_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def generate_training_data(self, n_samples=1000):
        """Generate synthetic weather data for training"""
        np.random.seed(42)
        
        temperature = np.random.normal(20, 15, n_samples)
        humidity = np.random.uniform(30, 95, n_samples)
        pressure = np.random.normal(1013, 20, n_samples)
        wind_speed = np.random.exponential(5, n_samples)
        
        conditions = []
        for i in range(n_samples):
            if temperature[i] > 25 and humidity[i] < 60:
                conditions.append('Sunny')
            elif humidity[i] > 80 and wind_speed[i] > 10:
                conditions.append('Rainy')
            elif temperature[i] < 10 or pressure[i] < 1000:
                conditions.append('Cloudy')
            else:
                conditions.append(np.random.choice(['Sunny', 'Cloudy', 'Rainy']))
        
        temperature += np.random.normal(0, 2, n_samples)
        humidity = np.clip(humidity + np.random.normal(0, 5, n_samples), 0, 100)
        
        return pd.DataFrame({
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'condition': conditions
        })
    
    def train_models(self):
        """Train the ML models"""
        print("Training weather prediction models...")
        
        # Generate training data
        data = self.generate_training_data(1000)
        
        # Prepare features
        X = data[['temperature', 'humidity', 'pressure', 'wind_speed']]
        y_condition = self.label_encoder.fit_transform(data['condition'])
        y_temp = data['temperature']
        y_humidity = data['humidity']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        self.condition_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.condition_model.fit(X_scaled, y_condition)
        
        self.temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.temp_model.fit(X_scaled, y_temp)
        
        self.humidity_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.humidity_model.fit(X_scaled, y_humidity)
        
        self.is_trained = True
        print("Models trained successfully!")
        
    def predict(self, temperature, humidity, pressure, wind_speed):
        """Make weather predictions"""
        if not self.is_trained:
            self.train_models()
        
        # Prepare input
        input_data = np.array([[temperature, humidity, pressure, wind_speed]])
        input_scaled = self.scaler.transform(input_data)
        
        # Make predictions
        condition_pred = self.condition_model.predict(input_scaled)[0]
        condition_name = self.label_encoder.inverse_transform([condition_pred])[0]
        condition_proba = self.condition_model.predict_proba(input_scaled)[0]
        
        temp_pred = self.temp_model.predict(input_scaled)[0]
        humidity_pred = self.humidity_model.predict(input_scaled)[0]
        
        return {
            'condition': condition_name,
            'condition_probabilities': dict(zip(self.label_encoder.classes_, condition_proba.tolist())),
            'predicted_temperature': float(temp_pred),
            'predicted_humidity': float(humidity_pred)
        }

# Initialize the weather API
weather_api = WeatherAPI()

@app.route('/')
def home():
    """Serve the HTML interface"""
    try:
        with open('weatherprediction.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <h1>Weather Prediction API</h1>
        <p>HTML file not found. Use the API endpoints:</p>
        <ul>
            <li>POST /predict - Make weather predictions</li>
            <li>GET /health - Check API health</li>
        </ul>
        """

@app.route('/predict', methods=['POST'])
def predict_weather():
    """API endpoint for weather prediction"""
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['temperature', 'humidity', 'pressure', 'wind_speed']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Make prediction
        result = weather_api.predict(
            float(data['temperature']),
            float(data['humidity']),
            float(data['pressure']),
            float(data['wind_speed'])
        )
        
        return jsonify({
            'success': True,
            'prediction': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_trained': weather_api.is_trained
    })

@app.route('/train', methods=['POST'])
def train_models():
    """Endpoint to retrain models"""
    try:
        weather_api.train_models()
        return jsonify({
            'success': True,
            'message': 'Models retrained successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("Starting Weather Prediction API...")
    print("Visit http://localhost:5000 to use the web interface")
    print("API endpoints:")
    print("  POST /predict - Make predictions")
    print("  GET /health - Health check")
    print("  POST /train - Retrain models")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
