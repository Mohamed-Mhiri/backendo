from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import json
import logging
import os

from nbeats_block import NBeatsBlock  # Ensure this is correctly imported from your project
from preprocessing import preprocess_data, prepare_data, denormalize, normalize_data  # Ensure these are correctly imported

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

# Load your model with a relative path
model_path = os.path.join(os.path.dirname(__file__), 'my_model.keras')
model = load_model(model_path, custom_objects={'NBeatsBlock': NBeatsBlock})
logging.info('Model loaded')

# Load normalization parameters with a relative path
normalization_params_path = os.path.join(os.path.dirname(__file__), 'normalization_params.json')
with open(normalization_params_path, 'r') as f:
    normalization_params = json.load(f)


min_values = normalization_params['min_values']
max_values = normalization_params['max_values']
mean_values = normalization_params['mean_values']
std_values = normalization_params['std_values']
logging.info('Normalization parameters loaded')

@app.route('/predict_week', methods=['POST'])
def predict_week():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400

        data = pd.read_csv(file)
        logging.debug(f'Uploaded data: \n{data.head()}')

        required_columns = ['timestamp', 'kwh', 'temperature_2m', 'relative_humidity_2m', 'weather_code',
                            'shortwave_radiation', 'is_day', 'wind_direction']
        if not all(column in data.columns for column in required_columns):
            missing_columns = [column for column in required_columns if column not in data.columns]
            return jsonify({'error': f'Missing required columns: {missing_columns}'}), 400

        data['timestamp'] = pd.to_datetime(data['timestamp'], dayfirst=True)
        processed_data = preprocess_data(data)
        logging.debug(f'Processed data: \n{processed_data.head()}')

        normalized_data = normalize_data(processed_data, min_values, max_values, mean_values, std_values)
        logging.debug(f'Normalized data: \n{normalized_data.head()}')

        X = normalized_data
        y = normalized_data['kwh']
        other_feats, windows_df, horizon = prepare_data(X, y)
        logging.debug(f'Initial windows_df: \n{windows_df.head()}')
        logging.debug(f'Initial other_feats: \n{other_feats.head()}')
        logging.debug(f'Horizon: \n{horizon.head()}')

        weekly_predictions = []

        if not isinstance(horizon.index, pd.DatetimeIndex):
            horizon.index = pd.to_datetime(horizon.index)

        future_timestamps = pd.date_range(start=horizon.index[-1], periods=24*7 + 1, freq='H')[1:]

        for i in range(24*7):
            predictions = model.predict([windows_df, other_feats])
            logging.debug(f'Raw model predictions: {predictions}')

            denormalized_predictions = denormalize(predictions, min_values, max_values, 'kwh').flatten()
            logging.debug(f'Denormalized predictions: {denormalized_predictions}')

            prediction_value = float(denormalized_predictions[0])
            if prediction_value < 0:
                prediction_value = 0
            weekly_predictions.append(prediction_value)

            logging.debug(f'Prediction step {i+1}: {prediction_value}')

            new_window_row = windows_df.iloc[-1].shift(-1)
            new_window_row[-1] = predictions[0][0]
            windows_df = pd.concat([windows_df.iloc[1:], pd.DataFrame([new_window_row], columns=windows_df.columns)])

            new_other_feats_row = other_feats.iloc[-1].copy()
            other_feats = pd.concat([other_feats.iloc[1:], pd.DataFrame([new_other_feats_row], columns=other_feats.columns)])

        # Denormalize the actual kwh values for comparison
        denormalized_horizon = denormalize(horizon.values, min_values, max_values, 'kwh')

        logging.debug(f"Length of horizon.index: {len(horizon.index)}")
        logging.debug(f"Length of future_timestamps: {len(future_timestamps)}")
        logging.debug(f"Length of weekly_predictions: {len(weekly_predictions)}")

        if len(future_timestamps) != len(weekly_predictions):
            raise ValueError("Length of future_timestamps and weekly_predictions do not match")

        return jsonify({
            'timestamps': horizon.index.tolist(),
            'actual_values': denormalized_horizon.tolist(),
            'future_timestamps': future_timestamps.tolist(),
            'predictions': weekly_predictions
        })

    except UnicodeDecodeError:
        return jsonify({'error': 'Invalid file format or encoding. Please ensure the file is in a supported format and encoding.'}), 400
    except Exception as e:
        logging.exception("An error occurred during prediction")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/')
def index():
    return "Welcome to the Energy Production Prediction API"

if __name__ == '__main__':
    app.run(debug=True)
