import time
import pandas as pd 
import numpy as np
from numpy.linalg import LinAlgError
import numpy.ma as ma
import matplotlib.pyplot as plt

def mvn_em(input_data, tol=1e-6, maxiter=100, plot=False):
    """
    Performs Expectation-Maximization (EM) for missing data imputation under a Multivariate Normal (MVN) model.

    Args:
        input_data (np.ndarray): The dataset with missing values. Shape (N, L), where N is the number of observations
                                 and L is the number of features.
        tol (float, optional): Convergence tolerance for stopping criteria. Defaults to 1e-6.
        maxiter (int, optional): Maximum number of iterations for the EM algorithm. Defaults to 100.
        plot (bool, optional): If True, plots the imputed values for different iterations of EM. Defaults to False.

    Returns:
        np.ndarray: The dataset with missing values imputed.
    """
    
    t1=time.time()

    Data = input_data.copy()
    N, L = Data.shape # Number of firms, Number of characteristics. 
    epsilon = 1e-5  # Small value for regularization
    
    mu = np.nanmean(Data, axis=0)
    Sigma = ma.cov(ma.masked_invalid(Data), rowvar=False)

    mask_nan = np.isnan(Data)

    # Initialization
    new_mu = mu.copy()
    new_Sigma = Sigma.copy()
    
    imputed_values = []

    update_mu=True
    for iter in range(maxiter):
        Adjusted_Sigma = np.zeros((L, L))
        lastNA = np.zeros(L, dtype=bool)

        # for each firm(i) in N, repeat:
        for i in range(N):
            NAcount = np.sum(mask_nan[i, :])
            this_mu = np.zeros(L)
            this_Sigma = np.zeros((L, L))
            thisNA = mask_nan[i, :]

            if NAcount == L:
                # When all data is missing, there's nothing to do. 
                this_mu = mu
                this_Sigma = Sigma
            elif NAcount == 0:
                # When data is complete, there's nothing to do.
                this_mu = Data[i, :]
                this_Sigma = np.zeros((L, L))
            else:
                if not np.array_equal(thisNA, lastNA):

                    R11 = Sigma[np.ix_(~thisNA, ~thisNA)]
                    R12 = Sigma[np.ix_(~thisNA, thisNA)]
                    R22 = Sigma[np.ix_(thisNA, thisNA)]
                    # print('\n R11 \n ', R11,'\n R12 \n ', R12, '\n R22 \n ', R22)
                    
                    # Regularize R11 to ensure it is invertible
                    R11 += np.eye(R11.shape[0]) * epsilon

                    # Solve R11 * X = R12
                    try:
                        invR11R12 = np.linalg.solve(R11, R12)
                        # print('\n invR11R12 \n ',invR11R12)
                    except LinAlgError:
                        raise ValueError("ERROR, SINGULAR MATRIX")

                    # R22 = R22 - R12.T * (R11 ^{-1}) * R12 # Adjust Sigma of unobserved values 
                    R22 = R22 - R12.T @ invR11R12
                    # print('\n adjusted R22 \n ', R22)

                E1 = Data[i, ~thisNA] - mu[~thisNA]
                E2 = mu[thisNA] + invR11R12.T @ E1
                # print('Nan imputed value using conditional expectation: \n ', E2)

                this_mu = Data[i, :]
                this_mu[thisNA] = E2           

                this_Sigma[np.ix_(thisNA, thisNA)] = R22
                # print(this_Sigma, '\n')
            
            this_Sigma += np.outer(this_mu, this_mu)
            # print(this_Sigma / N, '\n')

            Data[i, :] = this_mu
            Adjusted_Sigma += this_Sigma / N
            # print(Adjusted_Sigma)
            lastNA = thisNA

        if update_mu:
            new_mu = np.mean(Data, axis=0)


        new_Sigma = Adjusted_Sigma - np.outer(new_mu, new_mu)

        Edist = np.max(np.abs(new_mu - mu))
        Rdist = np.max(np.abs(new_Sigma - Sigma))

        # print(iter)
        if max(Edist, Rdist) > 1e15:
            raise ValueError("ERROR, DIVERGING")

        if Edist < tol and Rdist < tol:
            break

        mu = new_mu
        Sigma = new_Sigma
        # print(Data)
    
        imputed_values.append(Data[mask_nan])

    print('EM Total iterations: ', iter)
    
    t2 = time.time()
    print('EM Time in seconds:')
    print(t2-t1)


    # Plot imputations with respect to iterations.
    if plot == True:
        plt.figure(figsize=(12, 5))
        imputed_values_df = pd.DataFrame(imputed_values, 
                        columns=[f'Imputed_value{i+1}' for i in range(len(Data[mask_nan]))])
        imputed_values_df.plot()
        plt.title('Imputed Values for different iterations of EM')
        plt.xlabel('iteration')
        plt.ylabel('Imputed Value')
        plt.show()

    return Data



if __name__ == "__main__":
    # Example usage
    Data = np.array([
        [1.0, 2.0, np.nan],
        [2.0, np.nan, 3.0],
        [np.nan, 4.0, 5],
        [4.0, 5.0, 6.0]
    ])
    print("Original Matrix:")
    print(Data)

    imputed_data = mvn_em(Data, tol=1e-6, maxiter=100, plot=True)

    print("Imputed Data:")
    print(imputed_data)

