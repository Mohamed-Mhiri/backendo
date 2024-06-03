#preprocessing.py
import pandas as pd
import numpy as np
import json
import tensorflow as tf
import logging
# Load normalization parameters
with open('normalization_params.json', 'r') as f:
    normalization_params = json.load(f)

min_values = normalization_params['min_values']
max_values = normalization_params['max_values']
mean_values = normalization_params['mean_values']
std_values = normalization_params['std_values']
logging.info('Normalization parameters loaded')

def preprocess_data(data):
    """
    Preprocess the input data.

    Args:
    data (pd.DataFrame): Raw input data as a DataFrame.

    Returns:
    pd.DataFrame: Processed data ready for model prediction.
    """
    
    # Identify negative and NaN kWh values during nighttime
    nighttime_mask = (data['is_day'] == 0) & (data['kwh'].isna() | (data['kwh'] < 0))
    
    # Identify negative kWh values during the day
    daytime_negative_mask = (data['is_day'] == 1) & (data['kwh'] < 0)
    
    # Set negative and NaN kWh values to zero during nighttime
    data.loc[nighttime_mask, 'kwh'] = 0
    
    # Set negative kWh values during the day to NaN
    data.loc[daytime_negative_mask, 'kwh'] = np.nan
    
    # Interpolate missing values in the 'kwh' column
    data['kwh'] = data['kwh'].interpolate(method='linear')
    
    # Sort the dataset by timestamp
    data = data.sort_values(by='timestamp')
    
    # Reset the index
    data.set_index('timestamp', inplace=True)
    
    return data

def normalize_data(data, min_values, max_values, mean_values, std_values):
    numerical_features = ['kwh', 'relative_humidity_2m', 'shortwave_radiation']
    
    # Apply Min-Max scaling for numerical features
    for feature in numerical_features:
        data[feature] = (data[feature] - min_values[feature]) / (max_values[feature] - min_values[feature])
    
    # Apply zero-mean normalization for 'temperature_2m'
    data['temperature_2m'] = (data['temperature_2m'] - mean_values) / std_values
    
    # Apply sine and cosine transformation for wind direction
    data['wind_direction_sin'] = np.sin(2 * np.pi * data['wind_direction'] / 360)
    data['wind_direction_cos'] = np.cos(2 * np.pi * data['wind_direction'] / 360)
    
    # Drop the original wind direction column
    data.drop(columns=['wind_direction'], inplace=True)
    
    return data


def denormalize(data, min_values, max_values, feature):
    """
    Denormalize the input data for a specific feature using loaded parameters.

    Args:
    data (np.ndarray): Normalized data.
    min_values (pd.Series): Min values used for normalization.
    max_values (pd.Series): Max values used for normalization.
    feature (str): The feature name to denormalize.

    Returns:
    np.ndarray: Denormalized data.
    """
    if feature in ['kwh', 'relative_humidity_2m', 'shortwave_radiation']:
        return data * (max_values[feature] - min_values[feature]) + min_values[feature]
    elif feature == 'temperature_2m':
        return data * std_values[feature] + mean_values[feature]
    else:
        raise ValueError(f"Unknown feature: {feature}")


def prepare_data(features_df, target_series):
    """
    Prepare the data for the model.

    Args:
    features_df (pd.DataFrame): The feature DataFrame.
    target_series (pd.Series): The target series.

    Returns:
    tuple: Other features DataFrame, windows DataFrame, and the target horizon.
    """
    to_split = features_df.copy(deep=True)
    forecast = target_series
    to_split = to_split.shift(1)
    to_split.rename(columns={"kwh": "kwh_t_1"}, inplace=True)
    to_split['forecast'] = forecast
    windows = 24

    for n in range(2, windows + 1): 
        to_split[f'kwh_t_{n}'] = to_split['forecast'].shift(n)
        

    to_split.dropna(inplace=True)

    windows_df = to_split.filter(like='kwh_t_')
    other_feats = to_split.drop(columns=windows_df.columns.tolist() + ['forecast'])
    horizon = to_split['forecast']

    return other_feats, windows_df, horizon
