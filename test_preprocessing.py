# test_model_with_generated_data.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from nbeats_block import NBeatsBlock
from preprocessing import preprocess_data, normalize_data, prepare_data, denormalize
# Load your model
model = load_model("C:\\Users\\drago\\OneDrive\\Bureau\\pfe\\my_model.keras", custom_objects={'NBeatsBlock': NBeatsBlock})
print('Model loaded')

# Sample data generation function
def generate_sample_data():
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=240, freq='H').astype(str),
        'kwh': np.random.uniform(low=0, high=100, size=240),
        'temperature_2m': np.random.uniform(low=-10, high=35, size=240),
        'relative_humidity_2m': np.random.uniform(low=0, high=100, size=240),
        'weather_code': np.random.randint(low=0, high=10, size=240),
        'shortwave_radiation': np.random.uniform(low=0, high=1000, size=240),
        'is_day': np.random.randint(low=0, high=2, size=240),
        'wind_direction_100m': np.random.uniform(low=0, high=360, size=240)
    }
    return pd.DataFrame(data)

# Generate sample data
sample_data = generate_sample_data()

# Display the original data
print("Original Data:\n", sample_data)

# Process the data
processed_data = preprocess_data(sample_data)

# Normalize the data
normalized_data = normalize_data(processed_data)

print(normalized_data.shape)
print("normalized_data:\n", normalized_data)
# Split data into features and target
X = normalized_data
y = normalized_data['kwh']

model.summary()
# Prepare the data for time series prediction
other_feats, windows_df, horizon = prepare_data(X, y)
print("Shape of windows_df:", windows_df.shape)  # Should be (n_samples, 24)
print("Shape of other_feats:", other_feats.shape)  # Should be (n_samples, n_other_features)

# Make predictions
predictions = model.predict([windows_df, other_feats])
predictions_list = predictions.tolist()

# Denormalize predictions
denormalized_predictions = denormalize(np.array(predictions_list), 'kwh').tolist()


print("\nPredictions:")
print(denormalized_predictions)
