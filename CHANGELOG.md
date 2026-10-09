# Changelog

## version v0.2 Ridge Regression

the second version we changed the model from Linear Regression
to Ridge Regression to see if regularization could improve predictions

We also tested different values of alpha(0.1, 1, 10, 100) and used
5 fold cross-validation on the training data to select (alpha=10)

### the Results

| Version | Model | Test RMSE |
|---------|-------|-----------|
| v0.1 | LinearRegression | 53.8534 |
| v0.2 | Ridge (alpha=10) | 53.6263 |

The RMSE decreased by 0.2271 - thats about 0.42%.
This is not a big improvement but version v0.2 performed better on our
test split.

We tried different alpha values on the test set before
using cross validation

## version v0.1  (Baseline)

The first version used StandardScaler and LinearRegression

We used the scikit-learn diabetes dataset and split it into
80% training data and 20% testing data with random_state=42.
The baseline test RMSE was 53.8534.