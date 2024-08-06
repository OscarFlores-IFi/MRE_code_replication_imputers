import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Chen_McCoy_2024_ExpectationMaximization import mvn_em
from Xiong_Pelger_2023_factors_and_loadings import factor_loading_imputation

np.random.seed(42)
# Load the Iris dataset
Data = pd.read_csv('iris.csv')
original = Data.copy()

# Just for testing purpose;
# Add some missing values (Missing completely at random).
missing_mask = np.random.rand(Data.shape[0], Data.shape[1]) < 0.1
Data[missing_mask] = np.nan

# We will try to recover the original values. 
K = 2
imputed_data1, _ = factor_loading_imputation(Data.values, K)

print(imputed_data1)

imputed_data2 = mvn_em(Data.values)

print(imputed_data2)

difference1 = imputed_data1 - original
difference2 = imputed_data2 - original

# Create a DataFrame for better visualization with seaborn
diff1_df = pd.DataFrame(difference1)
diff2_df = pd.DataFrame(difference2)

vmin = min(np.nanmin(difference1), np.nanmin(difference2))
vmax = max(np.nanmax(difference1), np.nanmax(difference2))

# Plot the heatmaps
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(diff1_df, annot=False, cmap="coolwarm", center=0, vmin=vmin, vmax=vmax)
plt.title("Difference between Factor Loading Imputation and Original, K = {}".format(K))

plt.subplot(1, 2, 2)
sns.heatmap(diff2_df, annot=False, cmap="coolwarm", center=0, vmin=vmin, vmax=vmax)
plt.title("Difference between Expectation Maximization Imputation and Original")
plt.show()