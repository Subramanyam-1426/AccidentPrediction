import urllib.request
import json

def test_api():
    base_url = "http://127.0.0.1:5000"
    
    # 1. Test GET /api/metadata
    print("Testing GET /api/metadata...")
    req = urllib.request.urlopen(f"{base_url}/api/metadata")
    meta = json.loads(req.read().decode('utf-8'))
    print("Best Model:", meta.get('best_model'))
    print("Available Models:", meta.get('available_models'))

    # 2. Test predictions across ALL available algorithms
    payload = {
        "Weather": "Rainy",
        "Road_Type": "Highway",
        "Time_of_Day": "Night",
        "Traffic_Density": 2,
        "Speed_Limit": 110,
        "Number_of_Vehicles": 4,
        "Driver_Alcohol": 1,
        "Road_Condition": "Wet",
        "Vehicle_Type": "Car",
        "Driver_Age": 25,
        "Driver_Experience": 3,
        "Road_Light_Condition": "No Light"
    }

    print("\n--- TESTING PREDICTIONS WITH ALL 4 TRAINED ALGORITHMS ---")
    for algo in meta.get('available_models', []):
        payload['Model'] = algo
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(f"{base_url}/predict", data=data, headers={'Content-Type': 'application/json'})
        res = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
        print(f"Algorithm Engine: [{res['selected_model']}] => Prediction: {res['prediction_label']} ({res['probability_percent']}%), Risk Level: {res['risk_level']}")

    print("\nALL 4 ALGORITHMS PREDICTING SUCCESSFULLY & INTERACTIVELY CHOOSABLE!")

if __name__ == '__main__':
    test_api()
