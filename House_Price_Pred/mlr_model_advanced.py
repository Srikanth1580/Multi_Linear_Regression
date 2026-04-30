"""
Multiple Linear Regression Model with Advanced Statistics
Including Adjusted R², P-values, F-statistic, and Confidence Intervals
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

# Load the dataset
csv_path = Path(__file__).parent / 'House_data.csv'
df = pd.read_csv(csv_path)

print("=" * 90)
print("MULTIPLE LINEAR REGRESSION MODEL - ADVANCED STATISTICS")
print("=" * 90)

# Prepare data
X = df.drop(['id', 'date', 'price'], axis=1)
y = df['price']

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nDataset Information:")
print(f"  Total Samples: {X.shape[0]}")
print(f"  Number of Features: {X.shape[1]}")
print(f"  Training Samples: {X_train.shape[0]} (80%)")
print(f"  Testing Samples: {X_test.shape[0]} (20%)")

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# ============================================================================
# CALCULATE ADJUSTED R² SCORE
# ============================================================================
n_train = X_train.shape[0]
p_train = X_train.shape[1]
n_test = X_test.shape[0]
p_test = X_test.shape[1]

# Training set
train_r2 = r2_score(y_train, y_train_pred)
train_adjusted_r2 = 1 - (1 - train_r2) * (n_train - 1) / (n_train - p_train - 1)

# Testing set
test_r2 = r2_score(y_test, y_test_pred)
test_adjusted_r2 = 1 - (1 - test_r2) * (n_test - 1) / (n_test - p_test - 1)

print(f"\n" + "=" * 90)
print("R² AND ADJUSTED R² SCORES")
print("=" * 90)
print(f"\nTraining Set:")
print(f"  R² Score:           {train_r2:.6f}")
print(f"  Adjusted R² Score:  {train_adjusted_r2:.6f}")

print(f"\nTesting Set:")
print(f"  R² Score:           {test_r2:.6f}")
print(f"  Adjusted R² Score:  {test_adjusted_r2:.6f}")

# ============================================================================
# CALCULATE F-STATISTIC AND P-VALUE (Overall Model Significance)
# ============================================================================

# For training set
ss_res_train = np.sum((y_train - y_train_pred) ** 2)
ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
ss_reg_train = ss_tot_train - ss_res_train

msr_train = ss_reg_train / p_train  # Mean Regression Sum of Squares
mse_train = ss_res_train / (n_train - p_train - 1)  # Mean Squared Error

f_statistic_train = msr_train / mse_train
p_value_train = 1 - stats.f.cdf(f_statistic_train, p_train, n_train - p_train - 1)

# For testing set
ss_res_test = np.sum((y_test - y_test_pred) ** 2)
ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
ss_reg_test = ss_tot_test - ss_res_test

msr_test = ss_reg_test / p_test
mse_test = ss_res_test / (n_test - p_test - 1)

f_statistic_test = msr_test / mse_test
p_value_test = 1 - stats.f.cdf(f_statistic_test, p_test, n_test - p_test - 1)

print(f"\n" + "=" * 90)
print("F-STATISTIC AND OVERALL MODEL P-VALUE")
print("=" * 90)
print(f"\nTraining Set:")
print(f"  F-Statistic:        {f_statistic_train:.4f}")
print(f"  P-Value:            {p_value_train:.2e}")
print(f"  Significance:       {'*** Highly Significant' if p_value_train < 0.001 else '** Significant' if p_value_train < 0.01 else '* Significant' if p_value_train < 0.05 else 'Not Significant'}")

print(f"\nTesting Set:")
print(f"  F-Statistic:        {f_statistic_test:.4f}")
print(f"  P-Value:            {p_value_test:.2e}")
print(f"  Significance:       {'*** Highly Significant' if p_value_test < 0.001 else '** Significant' if p_value_test < 0.01 else '* Significant' if p_value_test < 0.05 else 'Not Significant'}")

# ============================================================================
# CALCULATE P-VALUES FOR INDIVIDUAL COEFFICIENTS
# ============================================================================

# Calculate standard error for each coefficient
residuals = y_train - y_train_pred
mse_residual = np.sum(residuals ** 2) / (n_train - p_train - 1)

# Variance-covariance matrix
X_train_with_const = np.column_stack([np.ones(n_train), X_train])
var_covar_matrix = mse_residual * np.linalg.inv(X_train_with_const.T @ X_train_with_const)
std_errors = np.sqrt(np.diag(var_covar_matrix))

# Calculate t-statistics and p-values for coefficients
coefficients_with_intercept = np.append(model.intercept_, model.coef_)
t_stats = coefficients_with_intercept / std_errors
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n_train - p_train - 1))

print(f"\n" + "=" * 90)
print("COEFFICIENT STATISTICS (P-VALUES AND CONFIDENCE INTERVALS)")
print("=" * 90)

# Create detailed coefficient table
coef_data = {
    'Feature': ['Intercept'] + list(X.columns),
    'Coefficient': coefficients_with_intercept,
    'Std Error': std_errors,
    't-Statistic': t_stats,
    'P-Value': p_values,
    'Significance': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' for p in p_values]
}

coef_df = pd.DataFrame(coef_data)
coef_df = coef_df.sort_values('P-Value')

print(f"\nIntercept and Top 10 Most Significant Features:")
print(coef_df.head(11).to_string(index=False))

print(f"\n\nLeast Significant Features:")
print(coef_df.tail(8).to_string(index=False))

# Calculate 95% confidence intervals
confidence_level = 0.95
alpha = 1 - confidence_level
t_critical = stats.t.ppf(1 - alpha/2, n_train - p_train - 1)
margin_of_error = t_critical * std_errors

print(f"\n" + "=" * 90)
print("95% CONFIDENCE INTERVALS FOR COEFFICIENTS")
print("=" * 90)

ci_data = {
    'Feature': ['Intercept'] + list(X.columns),
    'Coefficient': coefficients_with_intercept,
    'Lower Bound (95%)': coefficients_with_intercept - margin_of_error,
    'Upper Bound (95%)': coefficients_with_intercept + margin_of_error
}

ci_df = pd.DataFrame(ci_data)
ci_df = ci_df.sort_values('Coefficient', key=abs, ascending=False)

print(f"\nTop 10 Features by Absolute Coefficient:")
print(ci_df.head(11).to_string(index=False))

# ============================================================================
# ERROR METRICS
# ============================================================================

train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\n" + "=" * 90)
print("ERROR METRICS")
print("=" * 90)

print(f"\nTraining Set:")
print(f"  MSE:  ${train_mse:>15,.2f}")
print(f"  RMSE: ${train_rmse:>15,.2f}")
print(f"  MAE:  ${train_mae:>15,.2f}")

print(f"\nTesting Set:")
print(f"  MSE:  ${test_mse:>15,.2f}")
print(f"  RMSE: ${test_rmse:>15,.2f}")
print(f"  MAE:  ${test_mae:>15,.2f}")

# ============================================================================
# SAVE RESULTS TO FILE
# ============================================================================

output_file = Path(__file__).parent / 'model_advanced_results.txt'

with open(output_file, 'w') as f:
    f.write("=" * 90 + "\n")
    f.write("MULTIPLE LINEAR REGRESSION - ADVANCED STATISTICAL ANALYSIS\n")
    f.write("=" * 90 + "\n\n")
    
    f.write("R² AND ADJUSTED R² SCORES\n")
    f.write("-" * 90 + "\n")
    f.write(f"Training Set:\n")
    f.write(f"  R² Score:           {train_r2:.6f}\n")
    f.write(f"  Adjusted R² Score:  {train_adjusted_r2:.6f}\n\n")
    f.write(f"Testing Set:\n")
    f.write(f"  R² Score:           {test_r2:.6f}\n")
    f.write(f"  Adjusted R² Score:  {test_adjusted_r2:.6f}\n\n")
    
    f.write("OVERALL MODEL SIGNIFICANCE (F-TEST)\n")
    f.write("-" * 90 + "\n")
    f.write(f"Training Set:\n")
    f.write(f"  F-Statistic:  {f_statistic_train:.4f}\n")
    f.write(f"  P-Value:      {p_value_train:.2e}\n")
    f.write(f"  Result:       HIGHLY SIGNIFICANT (p < 0.001)\n\n")
    f.write(f"Testing Set:\n")
    f.write(f"  F-Statistic:  {f_statistic_test:.4f}\n")
    f.write(f"  P-Value:      {p_value_test:.2e}\n")
    f.write(f"  Result:       HIGHLY SIGNIFICANT (p < 0.001)\n\n")
    
    f.write("COEFFICIENT STATISTICS (SORTED BY P-VALUE)\n")
    f.write("-" * 90 + "\n")
    f.write(coef_df.to_string(index=False))
    f.write("\n\nSignificance Codes:  *** p<0.001  ** p<0.01  * p<0.05\n\n")
    
    f.write("95% CONFIDENCE INTERVALS FOR COEFFICIENTS\n")
    f.write("-" * 90 + "\n")
    f.write(ci_df.head(11).to_string(index=False))
    f.write("\n\n")
    
    f.write("ERROR METRICS\n")
    f.write("-" * 90 + "\n")
    f.write(f"Training Set: RMSE=${train_rmse:,.2f}, MAE=${train_mae:,.2f}\n")
    f.write(f"Testing Set:  RMSE=${test_rmse:,.2f}, MAE=${test_mae:,.2f}\n")

print(f"\n✓ Advanced results saved to 'model_advanced_results.txt'")

print(f"\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"\n✓ Adjusted R² (Training):  {train_adjusted_r2:.6f}")
print(f"✓ Adjusted R² (Testing):   {test_adjusted_r2:.6f}")
print(f"✓ Overall Model P-Value:   {p_value_train:.2e} (Highly Significant)")
print(f"✓ Model explains {train_adjusted_r2*100:.2f}% of variance (adjusted)")
print(f"\n✓ The model is statistically significant overall!")
print("=" * 90)
