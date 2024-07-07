import pytest
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

from data_processing import generate_features, split_data
from modelling import iterative_forecasting, optimize_lgbm

@pytest.fixture
def synthetic_data():
    # Create synthetic data for testing.
    data = pd.DataFrame({
        'id': range(1, 101),
        'date': pd.date_range(start='2021-01-01', periods=100, freq='D'),
        'country': np.random.choice(['A', 'B', 'C'], size=100),
        'store': np.random.choice(['X', 'Y', 'Z'], size=100),
        'product': np.random.choice(['P1', 'P2', 'P3'], size=100),
        'num_sold': np.random.randint(1, 100, size=100)
    })
    return data

def test_generate_features(synthetic_data):
    # Test feature generation.
    df_features = generate_features(synthetic_data)
    assert 'year' in df_features.columns
    assert 'month' in df_features.columns
    assert 'lag_1' in df_features.columns
    assert 'country_B' in df_features.columns or 'country_C' in df_features.columns
    assert 'store_Y' in df_features.columns or 'store_Z' in df_features.columns
    assert 'product_P2' in df_features.columns or 'product_P3' in df_features.columns

def test_split_data(synthetic_data):
    # Test the data splitting function.
    df_features = generate_features(synthetic_data)
    X_train, X_val, y_train, y_val, val_mask = split_data(df_features, validation_days=14)

    assert X_train.shape[0] + X_val.shape[0] == df_features.shape[0]
    assert y_train.shape[0] + y_val.shape[0] == df_features.shape[0]
    assert not val_mask.isnull().any()
    assert set(X_train.columns) == set(X_val.columns)

def test_optimize_lgbm(synthetic_data):
    # Test the LGBM hyperparameter optimization function.
    df_features = generate_features(synthetic_data)
    X_train, X_val, y_train, y_val, val_mask = split_data(df_features, validation_days=14)

    n_trials = 10
    date_col='date'
    cv = 5
    model = optimize_lgbm(X_train, y_train, date_col, n_trials, cv)

    assert isinstance(model, LGBMRegressor)

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    assert len(predictions) == len(y_val)


def test_iterative_forecasting(synthetic_data):
    # Test the iterative forecasting function.
    df_features = generate_features(synthetic_data)
    X_train, X_val, y_train, y_val, val_mask = split_data(df_features, validation_days=14)

    model = LGBMRegressor(random_state=42)
    df_forecast = iterative_forecasting(model, X_train, y_train, X_val, val_mask, df_features)

    assert df_forecast.shape[0] == X_val.shape[0]
    assert 'forecast' in df_forecast.columns
    assert not df_forecast['forecast'].isnull().any()