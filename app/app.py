from flask import Flask, request, jsonify, send_from_directory
import requests
import os


app = Flask(__name__)

# 🔧 Databricks Model Serving endpoint and token
DATABRICKS_URL = "https://dbc-a4864acb-d90e.cloud.databricks.com/serving-endpoints/nyc-taxi-fare/invocations"
# DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
DATABRICKS_TOKEN = "dapi252f160d2af154aa8f13a166fbbcc55d"


# Allow local web apps to call this backend
from flask_cors import CORS
CORS(app)

@app.route("/")
def home():
    # Serve your static HTML file
    return send_from_directory(app.static_folder, "index.html")


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        data = request.get_json()
        data = data.get("data")[0]
        trip_distance = float(data[0])
        is_rush_hour = int(data[1])

        if trip_distance <= 0:
                return jsonify({"error": "Invalid value: ", "trip distance": "must be > 0"}), 400

        payload = {
            "dataframe_split": {
                "columns": ["trip_distance", "is_rush_hour"],
                "data": [[trip_distance, is_rush_hour]]
            }
        }
        print(payload)

        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }

        response = requests.post(DATABRICKS_URL, headers=headers, json=payload)
        result = response.json()

        if response.status_code != 200:
            return jsonify({"error": "Databricks model error", "details": response.text}), response.status_code

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


TEST_PORT = 8080

if __name__ == "__main__":
    # Listen on 0.0.0.0 to accept external connections (required for Cloud Run)
    app.run(host='0.0.0.0', port=TEST_PORT)

