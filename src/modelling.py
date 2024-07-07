import pandas as pd
import numpy as np
import optuna
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from losses import smape, rmsle

smape_scorer = make_scorer(smape, greater_is_better=True)
rmsle_scorer = make_scorer(rmsle, greater_is_better=True)


def time_series_cv(X_train, y_train, date_col, cv=3):
    """
    Custom cross-validation for time series data.
    
    Parameters:
    - X_train (pd.DataFrame): training data (features), must include a date column
    - y_train (pd.Series): training data (target variable)
    - date_col (str): name of the date column
    - cv (int): number of folds for cross-validation (default is 3)
    
    Returns:
    - indices (list of tuples): list of (train_index, val_index) tuples
    """
    indices = []
    dates = np.sort(X_train[date_col].unique())
    n_samples = len(dates)
    for i in range(1, cv + 1):
        train_end_date = dates[n_samples - cv - 1]
        val_date = dates[n_samples - cv : n_samples - cv + i]
        train_indices = X_train[X_train['date'] <= train_end_date].index
        val_indices = X_train[X_train['date'].isin(val_date)].index
    
        if len(train_indices) > 0 and len(val_indices) > 0:
            indices.append((train_indices, val_indices))
    
    return indices

def optimize_lgbm(X_train, y_train, date_col, n_trials, cv=3):
    """
    Optimize hyperparameters for the LGBMRegressor model and return the tuned model.

    Parameters:
    - X_train (pd.DataFrame): training data (features), must include a date column
    - y_train (pd.Series): training data (target variable)
    - date_col (str): name of the date column
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
        
        cv_indices = time_series_cv(X_train, y_train, date_col, cv=cv)
        scores = []

        for train_idx, val_idx in cv_indices:
            X_train_fold, X_val_fold = X_train.loc[train_idx], X_train.loc[val_idx]
            y_train_fold, y_val_fold = y_train.loc[train_idx], y_train.loc[val_idx]
            
            model.fit(X_train_fold.drop(columns=[date_col]), y_train_fold)
            preds = model.predict(X_val_fold.drop(columns=[date_col]))
            score = smape_scorer._score_func(y_val_fold, preds)
            scores.append(score)
        
        return np.mean(scores)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    trial = study.best_trial
    best_params = trial.params
    
    model = LGBMRegressor(**best_params, random_state=42)
    return model



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

    unique_combinations = X_val.index.droplevel(-1).unique()

    for combination in unique_combinations:
        df_tmp_forecast = df_forecast.xs(combination, level=['country', 'store', 'product'], drop_level=False)
        
        for i in range(len(df_tmp_forecast)):
            idx = df_tmp_forecast.index[i]
            X_current = df_tmp_forecast.loc[idx, X_train.columns].values.reshape(1, -1)
            
            # Predict the current day's target value
            forecast = model.predict(X_current)[0]
            
            # Update the forecast in the dataframe
            df_forecast.at[idx, 'forecast'] = forecast
            
            # Update the features for the next day within the same group
            if i + 1 < len(df_tmp_forecast):
                next_idx = df_tmp_forecast.index[i + 1]
                for lag in range(1, 8):
                    lag_col = f'lag_{lag}'
                    if lag_col in df_tmp_forecast.columns:
                        if lag == 1:
                            df_forecast.at[next_idx, lag_col] = forecast
                        else:
                            prev_lag_col = f'lag_{lag - 1}'
                            df_forecast.at[next_idx, lag_col] = df_tmp_forecast.at[idx, prev_lag_col]

    return df_forecast