import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
import numpy.ma as ma
from numpy.linalg import eig
import time

def factor_loading_imputation(Data, K):
    """
    Performs imputation of missing values using factor loading approach.

    Args:
        Data (np.ndarray): The dataset with missing values. Shape (N, L), where N is the number of observations
                           and L is the number of features.
        K (int): Number of principal components (factors) to use for imputation. Must be less than or equal to L.

    Returns:
        Tuple:
            - np.ndarray: The dataset with missing values imputed.
            - np.ndarray: The imputed values only.
    """
    
    t1=time.time()

    N,L = Data.shape

    assert L >= K

    # Missing values 
    missing_mask = np.isnan(Data)
    Data[missing_mask] = np.nan

    # Standardize the observed data (mean=0, variance=1)
    mean_Z = np.nanmean(Data, axis=0).reshape(1, -1)  # Reshape to (1, L)
    std_Z = np.nanstd(Data, axis=0).reshape(1, -1)  # Reshape to (1, L)

    # Standardize the data
    Z_standardized = (Data - mean_Z) / std_Z
    Z_standardized[missing_mask] = np.nan

    # Compute the covariance matrix from the standardized data without missing values
    CovMatrix = ma.cov(ma.masked_invalid(Z_standardized), rowvar=False)
    # print(CovMatrix)


    # Perform eigen decomposition on the covariance matrix
    eigenvalues, eigenvectors = eig(CovMatrix)
    # print('\neigenvalues: ', eigenvalues)

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    top_eigenvectors = eigenvectors[:, :K]
    top_eigenvalues = eigenvalues[:K]

    # Scale the selected eigenvectors by sqrt(N) or by top eigenvalues (depending on the paper. Different scales are used.)
    scaled_eigenvectors = top_eigenvectors * top_eigenvalues

    # Perform regression using only the observed covariates.
    regression_results = []
    # print(Z_standardized)
    for i in range(N):
        sample = Z_standardized[i, :]
        # print(sample)
        # print(i)
        observed_mask = ~missing_mask[i, :]
        observed_values = sample[observed_mask]
        observed_eigenvectors = scaled_eigenvectors[observed_mask, :]
        reg = np.linalg.lstsq(observed_eigenvectors, observed_values, rcond=None)
        regression_results.append(reg[0])
        # print(reg)
        # print(reg)

    # Convert results to DataFrame
    regression_results_df = pd.DataFrame(regression_results, columns=[f'PC{i+1}' for i in range(K)])
    # print(regression_results_df.shape)

    # Estimate the values using only the selected eigenvectors
    Z_hat_standardized = regression_results_df.dot(scaled_eigenvectors.T)
    # print(Z_hat_standardized)

    # Transform Z_hat_standardized back to the original scale
    Z_hat = (Z_hat_standardized * std_Z) + mean_Z
    # print(Z_hat)

    Z_hat.iloc[~missing_mask] = Data[~missing_mask]

    imputed_values = Z_hat.values[missing_mask]
    # imputed_values.append(Data[mask_nan])
    # print(Z_hat)

    t2 = time.time()
    print('Factor_Loadings Time in seconds:')
    print(t2-t1)

    return Z_hat.values, imputed_values



if __name__ == "__main__":

    Data = np.array([
        [1.0, 2.0, np.nan],
        [2.0, np.nan, 3.0],
        [np.nan, 4.0, 5],
        [4.0, 5.0, 6.0]
    ])

    # Print results
    print("Original Matrix:")
    print(Data)

    list_imputed_values = []
    for K in range(1,4):
        print("With K = ", K)
        Z_hat, imputed_values = factor_loading_imputation(Data, K)

        list_imputed_values.append(imputed_values)
        print("Imputed Matrix:")
        print(Z_hat)

    imputed_values_df = pd.DataFrame(list_imputed_values, 
                            columns=[f'Imputed_value{i+1}' for i in range(len(imputed_values))], 
                            index = 1 + np.arange(len(list_imputed_values)))
    imputed_values_df.plot(figsize=(12, 5), marker = 'o')
    plt.title('Imputed Values for different Principal Components (K)')
    plt.xlabel('K')
    plt.ylabel('Imputed Value')
    plt.show()

