import numpy as np
import pandas as pd

def generate_features(df):
    """
    Generate features from the given dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns ['id', 'date', 'country', 'store', 'product', 'num_sold'].

    Returns
    ----------
    df : pandas.DataFrame
        DataFrame with generated features.
    """
    # Check if all required columns are present
    required_columns = ['id', 'date', 'country', 'store', 'product', 'num_sold']
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"

    # Ensure dates are in datetime format
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'])

    # Sort by date in ascending order
    df = df.sort_values('date')

    # Create features from 'date'
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(np.int64)

    # Creating lag features
    for lag in range(1, 8):  # Creating 7 lag features
        df[f'lag_{lag}'] = df['num_sold'].shift(lag)

    # Fill NA values caused by lag features
    df = df.fillna(method='bfill').fillna(method='ffill')

    # One-hot encoding with dtype=np.int64 to ensure 0/1 instead of True/False
    df = pd.get_dummies(df, columns=['country', 'store', 'product'], drop_first=True, dtype=np.int64)

    return df



def split_data(df, validation_days=14):
    """
    Split the dataframe into features (X) and target (y), and then into train and validation sets.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with generated features.
    validation_days : int
        Number of days to use for validation set.

    Returns
    ----------
    X_train : pandas.DataFrame
        Training features.
    X_val : pandas.DataFrame
        Validation features.
    y_train : pandas.Series
        Training target.
    y_val : pandas.Series
        Validation target.
    val_mask : pandas.Series
        Mask for validation set in the original dataframe.
    """
    # Get the unique dates in ascending order
    unique_dates = df['date'].sort_values().unique()
    
    # Determine the date threshold for validation set
    validation_start_date = unique_dates[-validation_days]
    
    # Split the data into train and validation sets based on date threshold
    train_mask = df['date'] < validation_start_date
    val_mask = df['date'] >= validation_start_date
    
    X_train = df[train_mask].drop(['id', 'date', 'num_sold'], axis=1)
    X_val = df[val_mask].drop(['id', 'date', 'num_sold'], axis=1)
    y_train = df.loc[train_mask, 'num_sold']
    y_val = df.loc[val_mask, 'num_sold']

    return X_train, X_val, y_train, y_val, val_mask