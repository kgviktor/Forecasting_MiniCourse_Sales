import pandas as pd
import numpy as np
import optuna
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from losses import smape, rmsle

smape_scorer = make_scorer(smape, greater_is_better=True)
rmsle_scorer = make_scorer(rmsle, greater_is_better=True)

def iterative_forecasting(model, X_train, y_train, X_val, val_mask, df):
    """
    Perform iterative forecasting on the validation set.

    Parameters
    ----------
    model : sklearn estimator
        The trained model to use for forecasting.
    X_train : pandas.DataFrame
        Training features.
    y_train : pandas.Series
        Training target.
    X_val : pandas.DataFrame
        Validation features.
    val_mask : pandas.Series
        Mask for validation set in the original dataframe.
    df : pandas.DataFrame
        Original dataframe with generated features.

    Returns
    ----------
    df_forecast : pandas.DataFrame
        DataFrame with forecasted values for the validation period.
    """
    df_forecast = df[val_mask].copy()
    df_forecast['forecast'] = np.nan

    # Train the model on the training set
    model.fit(X_train, y_train)

    # Perform iterative forecasting for each day in the validation set
    for i in range(len(df_forecast)):
        # Get the current day's features and drop the forecast column
        X_current = df_forecast.iloc[i][X_train.columns].values.reshape(1, -1)
        
        # Predict the current day's sales
        forecast = model.predict(X_current)[0]
        
        # Update the forecast in the dataframe
        df_forecast.iloc[i, df_forecast.columns.get_loc('forecast')] = forecast
        
        # Update the features for the next day
        if i + 1 < len(df_forecast):
            for lag in range(1, 8):
                if f'lag_{lag}' in df_forecast.columns:
                    df_forecast.iloc[i + 1, df_forecast.columns.get_loc(f'lag_{lag}')] = forecast if lag == 1 else df_forecast.iloc[i, df_forecast.columns.get_loc(f'lag_{lag - 1}')]

    return df_forecast



def optimize_lgbm(X_train, y_train, n_trials, cv=3):
    """
    Optimize hyperparameters for the LGBMRegressor model and return the tuned model.

    Parameters:
    - X_train (pd.DataFrame): training data (features)
    - y_train (pd.Series): training data (target variable)
    - n_trials (int): number of optimization iterations
    - cv (int): number of folds for cross-validation (default is 3)

    Returns:
    - model (LGBMRegressor): tuned model
    """
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'verbose': -1
        }

        model = LGBMRegressor(**param, random_state=42)
        
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring=smape_scorer).mean()
        return score

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    trial = study.best_trial
    best_params = trial.params
    
    model = LGBMRegressor(**best_params, random_state=42)
    return model