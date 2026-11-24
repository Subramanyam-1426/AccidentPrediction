import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from train_models import train_and_save

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

MODEL_FILE = 'all_models.pkl'
METRICS_FILE = 'model_metrics.json'
META_FILE = 'dataset_meta.json'
DATASET_FILE = 'dataset_traffic_accident_prediction1.csv'

# Ensure model and metadata exist
if not (os.path.exists(MODEL_FILE) and os.path.exists(METRICS_FILE) and os.path.exists(META_FILE)):
    print("Models or metadata missing. Running training pipeline...")
    train_and_save()

# Load all model pipelines
models_dict = joblib.load(MODEL_FILE)
pipelines = models_dict['pipelines']
best_model_name = models_dict['best_model']

with open(METRICS_FILE, 'r') as f:
    metrics_data = json.load(f)

with open(META_FILE, 'r') as f:
    dataset_meta = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    return jsonify({
        'best_model': best_model_name,
        'available_models': list(pipelines.keys()),
        'metadata': dataset_meta
    })

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify(metrics_data)

@app.route('/api/dashboard-data', methods=['GET'])
def get_dashboard_data():
    if not os.path.exists(DATASET_FILE):
        return jsonify({'error': 'Dataset file not found'}), 404
        
    df = pd.read_csv(DATASET_FILE)
    df_clean = df.dropna(subset=['Accident'])
    
    target_counts = df_clean['Accident'].value_counts().to_dict()
    target_dist = {
        'No Accident': int(target_counts.get(0.0, 0)),
        'Accident': int(target_counts.get(1.0, 0))
    }
    
    def get_accident_rate(column_name, label_map=None):
        if column_name not in df_clean.columns:
            return {}
        grouped = df_clean.groupby(column_name)['Accident'].agg(['count', 'sum']).reset_index()
        result = {}
        for _, row in grouped.iterrows():
            val = row[column_name]
            if label_map and val in label_map:
                key = label_map[val]
            else:
                key = str(val)
            result[key] = {
                'total': int(row['count']),
                'accidents': int(row['sum']),
                'no_accidents': int(row['count'] - row['sum']),
                'rate': round(float(row['sum'] / row['count'] * 100), 1) if row['count'] > 0 else 0.0
            }
        return result

    traffic_density_map = {0.0: 'Low', 1.0: 'Medium', 2.0: 'High'}
    alcohol_map = {0.0: 'No', 1.0: 'Yes'}
    
    by_weather = get_accident_rate('Weather')
    by_road_type = get_accident_rate('Road_Type')
    by_road_condition = get_accident_rate('Road_Condition')
    by_traffic_density = get_accident_rate('Traffic_Density', traffic_density_map)
    by_time_of_day = get_accident_rate('Time_of_Day')
    by_vehicle_type = get_accident_rate('Vehicle_Type')
    by_lighting = get_accident_rate('Road_Light_Condition')
    by_alcohol = get_accident_rate('Driver_Alcohol', alcohol_map)
    
    return jsonify({
        'total_samples': len(df_clean),
        'target_distribution': target_dist,
        'by_weather': by_weather,
        'by_road_type': by_road_type,
        'by_road_condition': by_road_condition,
        'by_traffic_density': by_traffic_density,
        'by_time_of_day': by_time_of_day,
        'by_vehicle_type': by_vehicle_type,
        'by_lighting': by_lighting,
        'by_alcohol': by_alcohol
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    # User chosen model algorithm (defaults to best_model)
    selected_model_name = data.get('Model', best_model_name)
    if selected_model_name not in pipelines:
        selected_model_name = best_model_name

    active_pipeline = pipelines[selected_model_name]

    # Extract inputs
    weather = data.get('Weather')
    road_type = data.get('Road_Type')
    time_of_day = data.get('Time_of_Day')
    traffic_density = data.get('Traffic_Density')
    speed_limit = data.get('Speed_Limit')
    number_of_vehicles = data.get('Number_of_Vehicles')
    driver_alcohol = data.get('Driver_Alcohol')
    road_condition = data.get('Road_Condition')
    vehicle_type = data.get('Vehicle_Type')
    driver_age = data.get('Driver_Age')
    driver_experience = data.get('Driver_Experience')
    road_light_condition = data.get('Road_Light_Condition')

    # Data conversion & normalization
    try:
        speed_limit = float(speed_limit)
        number_of_vehicles = float(number_of_vehicles)
        driver_age = float(driver_age)
        driver_experience = float(driver_experience)
        
        if isinstance(traffic_density, str):
            td_lower = traffic_density.lower()
            if td_lower == 'low':
                traffic_density = 0.0
            elif td_lower == 'medium':
                traffic_density = 1.0
            elif td_lower == 'high':
                traffic_density = 2.0
            else:
                traffic_density = float(traffic_density)
        else:
            traffic_density = float(traffic_density)

        if isinstance(driver_alcohol, str):
            driver_alcohol = 1.0 if driver_alcohol.lower() in ['yes', '1', 'true'] else 0.0
        else:
            driver_alcohol = float(driver_alcohol)

    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid numerical input formats: {str(e)}'}), 400

    # Strict Validation Rules
    validation_errors = []
    if driver_age < 18 or driver_age > 80:
        validation_errors.append("Driver age must be between 18 and 80 years.")
    if driver_experience < 0 or driver_experience > 60:
        validation_errors.append("Driver experience must be between 0 and 60 years.")
    if driver_experience > (driver_age - 16):
        validation_errors.append(f"Driver experience ({int(driver_experience)} yrs) cannot exceed realistic driving age limit (Age {int(driver_age)} - 16 = {int(driver_age - 16)} yrs).")
    if number_of_vehicles < 1:
        validation_errors.append("Number of vehicles must be at least 1.")
    if speed_limit < 10 or speed_limit > 250:
        validation_errors.append("Speed limit must be within 10 to 250 km/h.")

    if validation_errors:
        return jsonify({'error': 'Validation Failed', 'details': validation_errors}), 400

    input_dict = {
        'Traffic_Density': [traffic_density],
        'Speed_Limit': [speed_limit],
        'Number_of_Vehicles': [number_of_vehicles],
        'Driver_Alcohol': [driver_alcohol],
        'Driver_Age': [driver_age],
        'Driver_Experience': [driver_experience],
        'Weather': [weather],
        'Road_Type': [road_type],
        'Time_of_Day': [time_of_day],
        'Road_Condition': [road_condition],
        'Vehicle_Type': [vehicle_type],
        'Road_Light_Condition': [road_light_condition]
    }
    
    input_df = pd.DataFrame(input_dict)

    # Perform prediction using chosen pipeline
    pred_class = int(active_pipeline.predict(input_df)[0])
    
    if hasattr(active_pipeline, "predict_proba"):
        probabilities = active_pipeline.predict_proba(input_df)[0]
        prob_accident = float(probabilities[1])
    else:
        prob_accident = 1.0 if pred_class == 1 else 0.0

    prob_percent = round(prob_accident * 100, 1)

    if prob_percent < 40:
        risk_level = "LOW"
        risk_color = "emerald"
        prediction_label = "NO ACCIDENT LIKELY"
    elif prob_percent < 70:
        risk_level = "MODERATE"
        risk_color = "amber"
        prediction_label = "ACCIDENT LIKELY" if pred_class == 1 else "NO ACCIDENT LIKELY"
    else:
        risk_level = "HIGH"
        risk_color = "rose"
        prediction_label = "ACCIDENT LIKELY"

    td_labels = {0.0: 'Low', 1.0: 'Medium', 2.0: 'High'}
    key_conditions = [
        {"icon": "🌧️", "label": "Weather", "value": str(weather)},
        {"icon": "🛣️", "label": "Road Type", "value": str(road_type)},
        {"icon": "🛣️", "label": "Road Condition", "value": str(road_condition)},
        {"icon": "🚗", "label": "Vehicles", "value": int(number_of_vehicles)},
        {"icon": "🍺", "label": "Driver Alcohol", "value": "Yes" if driver_alcohol == 1.0 else "No"},
        {"icon": "💡", "label": "Lighting", "value": str(road_light_condition)},
        {"icon": "🚦", "label": "Traffic Density", "value": td_labels.get(traffic_density, str(traffic_density))},
        {"icon": "⚡", "label": "Speed Limit", "value": f"{int(speed_limit)} km/h"}
    ]

    recommendations = []
    if driver_alcohol == 1.0:
        recommendations.append("🍺 Never drive under the influence of alcohol. Designate a sober driver or use rideshare services.")
    if str(weather).lower() in ['rainy', 'snowy', 'foggy', 'stormy']:
        recommendations.append("🌧️ Adverse weather detected. Reduce your driving speed below the posted limit and turn on fog/low-beam headlights.")
    if str(road_condition).lower() in ['wet', 'icy', 'under construction']:
        recommendations.append("⚠️ Slippery or compromised road condition. Increase braking distance to at least 4-5 car lengths.")
    if traffic_density >= 2.0:
        recommendations.append("🚦 High traffic density. Maintain a safe following distance and avoid abrupt lane changes.")
    if str(road_light_condition).lower() in ['no light', 'artificial light', 'night'] or str(time_of_day).lower() == 'night':
        recommendations.append("💡 Reduced visibility conditions. Ensure headlights are clean and maintain heightened vigilance for hazards.")
    if speed_limit > 100:
        recommendations.append("⚡ High speed corridor. Observe speed limits strictly and stay alert for sudden traffic slow-downs.")
    if driver_experience < 3:
        recommendations.append("🔰 Novice driver risk factor. Avoid non-essential night driving or hazardous weather conditions.")

    if risk_level == "HIGH":
        recommendations.insert(0, "🚨 HIGH-RISK DRIVING ENVIRONMENT: Strongly consider delaying your journey until weather, light, or traffic conditions improve.")
    elif risk_level == "MODERATE":
        recommendations.insert(0, "⚠️ MODERATE RISK DETECTED: Proceed with extra caution and strictly observe all road safety guidelines.")
    else:
        recommendations.insert(0, "🟢 SAFE CONDITIONS: Conditions appear favorable, but always keep your seatbelt fastened and stay alert.")

    top_features = metrics_data['metrics'].get(selected_model_name, {}).get('top_features', {})

    return jsonify({
        'prediction': pred_class,
        'prediction_label': prediction_label,
        'probability': round(prob_accident, 4),
        'probability_percent': prob_percent,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'selected_model': selected_model_name,
        'best_model_default': best_model_name,
        'key_conditions': key_conditions,
        'recommendations': recommendations,
        'model_top_features': top_features
    })

if __name__ == '__main__':
    print("Starting Accident Prediction System API Server on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
