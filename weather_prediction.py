import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class WeatherPredictor:
    def __init__(self):
        self.condition_model = None
        self.temp_model = None
        self.humidity_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic weather data for demonstration"""
        np.random.seed(42)
        
        # Generate base features
        temperature = np.random.normal(20, 15, n_samples)  # Celsius
        humidity = np.random.uniform(30, 95, n_samples)    # Percentage
        pressure = np.random.normal(1013, 20, n_samples)   # hPa
        wind_speed = np.random.exponential(5, n_samples)   # km/h
        
        # Create realistic relationships
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
        
        # Add some noise to make it more realistic
        temperature += np.random.normal(0, 2, n_samples)
        humidity = np.clip(humidity + np.random.normal(0, 5, n_samples), 0, 100)
        
        data = pd.DataFrame({
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'condition': conditions
        })
        
        return data
    
    def prepare_data(self, data):
        """Prepare data for training"""
        # Features for prediction
        feature_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
        X = data[feature_cols]
        
        # Encode weather conditions
        y_condition = self.label_encoder.fit_transform(data['condition'])
        y_temp = data['temperature']
        y_humidity = data['humidity']
        
        return X, y_condition, y_temp, y_humidity
    
    def train_models(self, data):
        """Train both classification and regression models"""
        X, y_condition, y_temp, y_humidity = self.prepare_data(data)
        
        # Split data
        X_train, X_test, y_cond_train, y_cond_test = train_test_split(
            X, y_condition, test_size=0.2, random_state=42
        )
        _, _, y_temp_train, y_temp_test = train_test_split(
            X, y_temp, test_size=0.2, random_state=42
        )
        _, _, y_hum_train, y_hum_test = train_test_split(
            X, y_humidity, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train weather condition classifier
        self.condition_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.condition_model.fit(X_train_scaled, y_cond_train)
        
        # Train temperature regressor
        self.temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.temp_model.fit(X_train_scaled, y_temp_train)
        
        # Train humidity regressor
        self.humidity_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.humidity_model.fit(X_train_scaled, y_hum_train)
        
        # Evaluate models
        self.evaluate_models(X_test_scaled, y_cond_test, y_temp_test, y_hum_test)
        
        return X_train_scaled, X_test_scaled, y_cond_train, y_cond_test
    
    def evaluate_models(self, X_test, y_cond_test, y_temp_test, y_hum_test):
        """Evaluate model performance"""
        # Weather condition prediction
        cond_pred = self.condition_model.predict(X_test)
        cond_accuracy = accuracy_score(y_cond_test, cond_pred)
        
        # Temperature prediction
        temp_pred = self.temp_model.predict(X_test)
        temp_mse = mean_squared_error(y_temp_test, temp_pred)
        temp_r2 = r2_score(y_temp_test, temp_pred)
        
        # Humidity prediction
        hum_pred = self.humidity_model.predict(X_test)
        hum_mse = mean_squared_error(y_hum_test, hum_pred)
        hum_r2 = r2_score(y_hum_test, hum_pred)
        
        print("=== MODEL EVALUATION ===")
        print(f"Weather Condition Accuracy: {cond_accuracy:.3f}")
        print(f"Temperature MSE: {temp_mse:.3f}, R²: {temp_r2:.3f}")
        print(f"Humidity MSE: {hum_mse:.3f}, R²: {hum_r2:.3f}")
        
        # Classification report
        print("\nWeather Condition Classification Report:")
        print(classification_report(y_cond_test, cond_pred, 
                                  target_names=self.label_encoder.classes_))
    
    def predict_weather(self, temperature, humidity, pressure, wind_speed):
        """Make predictions for new weather data"""
        if not all([self.condition_model, self.temp_model, self.humidity_model]):
            raise ValueError("Models not trained yet!")
        
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
            'condition_probabilities': dict(zip(self.label_encoder.classes_, condition_proba)),
            'predicted_temperature': temp_pred,
            'predicted_humidity': humidity_pred
        }
    
    def plot_feature_importance(self):
        """Plot feature importance for the models"""
        feature_names = ['Temperature', 'Humidity', 'Pressure', 'Wind Speed']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Weather condition model
        axes[0].bar(feature_names, self.condition_model.feature_importances_)
        axes[0].set_title('Weather Condition Model\nFeature Importance')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Temperature model
        axes[1].bar(feature_names, self.temp_model.feature_importances_)
        axes[1].set_title('Temperature Model\nFeature Importance')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Humidity model
        axes[2].bar(feature_names, self.humidity_model.feature_importances_)
        axes[2].set_title('Humidity Model\nFeature Importance')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Initialize predictor
    predictor = WeatherPredictor()
    
    # Generate sample data
    print("Generating sample weather data...")
    weather_data = predictor.generate_sample_data(1000)
    
    print("\nDataset Info:")
    print(weather_data.head())
    print(f"\nDataset shape: {weather_data.shape}")
    print(f"Weather conditions: {weather_data['condition'].value_counts().to_dict()}")
    
    # Train models
    print("\nTraining models...")
    predictor.train_models(weather_data)
    
    # Plot feature importance
    predictor.plot_feature_importance()
    
    # Example predictions
    print("\n=== EXAMPLE PREDICTIONS ===")
    
    test_cases = [
        (30, 45, 1020, 5),    # Hot, low humidity
        (15, 85, 995, 15),    # Cool, high humidity, low pressure
        (22, 65, 1013, 8),    # Moderate conditions
    ]
    
    for i, (temp, hum, press, wind) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: T={temp}°C, H={hum}%, P={press}hPa, W={wind}km/h")
        result = predictor.predict_weather(temp, hum, press, wind)
        print(f"Predicted Condition: {result['condition']}")
        print(f"Predicted Temperature: {result['predicted_temperature']:.1f}°C")
        print(f"Predicted Humidity: {result['predicted_humidity']:.1f}%")
        print("Condition Probabilities:")
        for condition, prob in result['condition_probabilities'].items():
            print(f"  {condition}: {prob:.3f}")

if __name__ == "__main__":
    main()
