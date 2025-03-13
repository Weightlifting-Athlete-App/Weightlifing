from flask import Flask, request, jsonify
from dbConfig import get_db_connection
from bson import ObjectId
import json
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer

# Initialize Flask app
app = Flask(__name__)

# Get MongoDB connection
db, collection = get_db_connection()

# Load the model and scaler
model = joblib.load('stacking_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load performance scaler if it was used
try:
    performance_scaler = joblib.load("performance_scaler.pkl")
    is_performance_scaled = True
except FileNotFoundError:
    is_performance_scaled = False

# Custom JSON encoder to handle ObjectId
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

# Function to preprocess user data
def preprocess_user_data(user_data):
    flattened_data = {
        "age": user_data["age"],
        "age_start": user_data["age_start"],
        "yrs_experience": user_data["yrs_experience"],
        "sex_encoded": user_data["sex_encoded"],
        "body_weight": user_data["body_weight"],
        "lifted_weight": user_data["lifted_weight"],
        "shoulder_angle": user_data["pose_data"]["angles"]["shoulder_angle"],
        "knees_angle": user_data["pose_data"]["angles"]["knees_angle"],
        "back_angle": user_data["pose_data"]["angles"]["back_angle"],
        "wrist_angle": user_data["pose_data"]["angles"]["wrist_angle"],
        "hips_angle": user_data["pose_data"]["angles"]["hips_angle"]
    }
    
    return pd.DataFrame([flattened_data])

# Function to evaluate performance
def evaluate_performance(predicted_performance):
    if predicted_performance >= 120:
        return "Excellent"
    elif 120 <= predicted_performance < 120:
        return "Good"
    elif 100 <= predicted_performance < 100:
        return "Average"
    else:
        return "Needs Improvement"

# Prediction function
def predict_user_input(user_data):
    df = preprocess_user_data(user_data)

    # Debugging: Print incoming data before scaling
    print("\n📌 Data Before Scaling:\n", df)

    # Check if all expected features are present
    expected_features = scaler.feature_names_in_
    if set(expected_features) != set(df.columns):
        missing_features = set(expected_features) - set(df.columns)
        extra_features = set(df.columns) - set(expected_features)
        raise ValueError(f"Missing Features: {missing_features}, Unexpected Features: {extra_features}")

    # Scale the input data
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

    # Debugging: Print data after scaling
    print("\n📌 Data After Scaling:\n", df_scaled)

    # Predict Performance (scaled value)
    predicted_scaled = model.predict(df_scaled)[0]

    # 🔄 Convert back to original scale if performance was scaled
    if is_performance_scaled:
        predicted_performance = performance_scaler.inverse_transform([[predicted_scaled]])[0][0]
    else:
        predicted_performance = predicted_scaled

    # Debugging: Show predicted value
    print(f"\n🎯 Predicted Performance (Original Scale): {predicted_performance:.4f}")

    # Evaluate Performance
    performance_category = evaluate_performance(predicted_performance)
    print(f"🏆 Performance Evaluation: {performance_category}")

    return predicted_performance, performance_category

# API endpoint to collect user data and pose data
@app.route("/submit_user_data", methods=["POST"])
def submit_user_data():
    user_data = request.json
    if not user_data or "username" not in user_data:
        return jsonify({"error": "Username is required"}), 400

    print(f"\n📩 Received Request for: {user_data['username']}")

    if collection is not None:
        try:
            # Predict performance and evaluate it
            predicted_performance, performance_category = predict_user_input(user_data)
            user_data["predicted_performance"] = predicted_performance
            user_data["performance_category"] = performance_category  # Store evaluation

            # Store user data in MongoDB
            result = collection.insert_one(user_data)
            user_data["_id"] = str(result.inserted_id)
            print(f"✅ Data Stored in MongoDB for {user_data['username']}")

            return jsonify({
                "message": "User data stored successfully!",
                "data": user_data
            }), 201

        except Exception as e:
            print(f"❌ Prediction Error: {e}")
            return jsonify({"error": f"Prediction failed: {e}"}), 500

    else:
        return jsonify({"error": "Failed to connect to MongoDB"}), 500

if __name__ == "__main__":
    app.run(debug=True)
