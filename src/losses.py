import numpy as np

# SMAPE function
def smape(A, F):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).

    SMAPE is an accuracy measure based on percentage (or relative) errors.
    It is usually defined as follows:
    
    SMAPE = 100/n * sum(2 * abs(F - A) / (abs(A) + abs(F)))
    
    where:
    - A - true values
    - F - forecasted values
    - n - number of observations

    Parameters
    ----------
    A : numpy.ndarray
        Array of true values.
    F : numpy.ndarray
        Array of forecasted values.

    Returns
    ----------
    float
        SMAPE value as a percentage.

    Examples
    ----------
    >>> A = np.array([100, 200, 300])
    >>> F = np.array([110, 190, 310])
    >>> smape(A, F)
    4.761904761904762
    """
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))



# RMSLE function
def rmsle(A, F):
    """
    Calculates the Root Mean Squared Logarithmic Error (RMSLE).

    RMSLE is an accuracy measure based on the logarithmic difference between 
    the predicted and true values.
    
    It is usually defined as follows:
    
    RMSLE = sqrt(1/n * sum((log(F + 1) - log(A + 1))^2))
    
    where:
    - A - true values
    - F - forecasted values
    - n - number of observations

    Parameters
    ----------
    A : numpy.ndarray
        Array of true values.
    F : numpy.ndarray
        Array of forecasted values.

    Returns
    ----------
    float
        RMSLE value.

    Examples
    ----------
    >>> A = np.array([100, 200, 300])
    >>> F = np.array([110, 190, 310])
    >>> rmsle(A, F)
    0.0434
    """
    return np.sqrt(np.mean((np.log1p(F) - np.log1p(A))**2))