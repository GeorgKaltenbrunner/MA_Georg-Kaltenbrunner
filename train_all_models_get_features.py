# Imports
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# Read Data
df_train = pd.read_csv(r'80_train.csv')

# Utilization
uti = 80

# Global model_performance_df
model_performance_df = pd.DataFrame()
features_dict = dict()


# Model performance function
def model_performance(model, pt, feature, features, model_name, X, y, y_pred, y_log, y_pred_log):
    global model_performance_df
    r2 = r2_score(y, y_pred)
    n = X.shape[0]
    p = X.shape[1]
    r2_adj = 1 - (((1 - r2) * (n - 1)) / (n - p - 1))
    if 'exp' in model:
        aic = n * np.log(mean_squared_error(y_log, y_pred_log)) + 2 * p
        aic_non_lin = np.log10(mean_squared_error(y, y_pred)) + 2 * p / n
    else:
        aic = n * np.log(mean_squared_error(y, y_pred)) + 2 * p
        aic_non_lin = 0
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    features = features.values

    # new df
    new_df = pd.DataFrame(
        {'model': model, 'product_type': pt, 'search': feature, 'features': [features], 'model_name': model_name,
         'r2': r2, 'r2_adjusted': r2_adj, 'aic': aic, 'aic_non_lin': aic_non_lin, 'mae': mae,
         'mse': mse, 'rmse': rmse}, index=[uti])
    model_performance_df = pd.concat([model_performance_df, new_df])


# Linear Regression
def feature_lin_regression(X, y, pt, search_direction):
    """
    :param pt: Product type
    :param X: X train values.
    :param y: y train values.
    :param search_direction: forward vs backward for search direction
    :return: Performance of model.
    """

    pipe = Pipeline([('scaler', MinMaxScaler()),
                     ('sfs',
                      SFS(LinearRegression(), direction=search_direction, scoring='neg_mean_squared_error', cv=5)),
                     ('lr', LinearRegression())])

    pipe.fit(X, y)

    # Select features
    selected_features = X.columns[pipe.named_steps['sfs'].support_]
    features_dict[(f"{uti}_linear_regression_{search_direction}")] = dict()
    features_dict[(f"{uti}_linear_regression_{search_direction}")][pt] = selected_features.tolist()

    # Train Model on selected Features
    pipe_model = Pipeline([('scaler', MinMaxScaler()), ('reg', LinearRegression())])
    X_selected = X[selected_features]
    pipe_model.fit(X_selected, y)

    y_pred = pipe_model.predict(X_selected)

    # Save model
    model_name = (f"{uti}_{pt}_linear_regression_{search_direction}.sav")

    # Track performance
    model = 'linear_regression'
    feature = search_direction
    model_performance(model, pt, feature, selected_features, model_name, X_selected, y, y_pred, 1, 1)


# Linear Regression RFECV
def rfecv_feature_lin_regression(X, y, pt):
    """
    :param pt: Product type
    :param X: X train values.
    :param y: y train values.
    :return: Performance of model.
    """

    pipe = Pipeline([('scaler', MinMaxScaler()),
                     ('rfecv', RFECV(LinearRegression(), step=1, cv=5)),
                     ('lr', LinearRegression())])

    pipe.fit(X, y)

    # Select features
    selected_features = X.columns[pipe.named_steps['rfecv'].support_]
    features_dict[(f"{uti}_linear_regression_rfecv")] = dict()
    features_dict[(f"{uti}_linear_regression_rfecv")][pt] = selected_features.tolist()

    # Train Model on selected Features
    pipe_model = Pipeline([('scaler', MinMaxScaler()), ('reg', LinearRegression())])
    X_selected = X[selected_features]
    pipe_model.fit(X_selected, y)

    y_pred = pipe_model.predict(X_selected)

    # Save model
    model_name = (f"{uti}_{pt}_linear_regression_rfecv.sav")

    # Track performance
    model = 'linear_regression'
    feature = 'rfecv'
    model_performance(model, pt, feature, selected_features, model_name, X_selected, y, y_pred, 1, 1)


# Exp Regression using Lin Regression

def feature_exp_regression(X, y, pt, search_direction):
    """
    :param pt: Product type
    :param X: X train values.
    :param y: y train values.
    :param search_direction: forward vs backward for search direction
    :return: Performance of model.
    """

    # Transform y zu y_log
    y_log = np.log(y)
    mean_log = np.median(y_log)
    y_log = np.where(y_log == np.inf, mean_log, y_log)
    y_log = np.where(y_log == -np.inf, mean_log, y_log)

    pipe = Pipeline([('scaler', MinMaxScaler()),
                     ('sfs',
                      SFS(LinearRegression(), direction=search_direction, scoring='neg_mean_squared_error', cv=5)),
                     ('regressor', LinearRegression())])

    pipe.fit(X, y_log)

    # Select features
    selected_features = X.columns[pipe.named_steps['sfs'].support_]
    features_dict[(f"{uti}_exp_regression_{search_direction}")] = dict()
    features_dict[(f"{uti}_exp_regression_{search_direction}")][pt] = selected_features.tolist()

    # Train Model on selected Features
    pipe_model = Pipeline([('scaler', MinMaxScaler()), ('reg', LinearRegression())])
    X_selected = X[selected_features]
    pipe_model.fit(X_selected, y_log)

    y_pred_log = pipe_model.predict(X_selected)
    y_pred = np.exp(y_pred_log)

    # Save model
    model_name = (f"{uti}_{pt}_exp_regression_{search_direction}.sav")

    # Track performance
    model = 'exp_regression'
    feature = search_direction
    model_performance(model, pt, feature, selected_features, model_name, X_selected, y, y_pred, y_log, y_pred_log)


# Exp Regression using Lin Regression rfecv

def rfecv_feature_exp_regression(X, y, pt):
    """
    :param pt: Product type
    :param X: X train values.
    :param y: y train values.
    :return: Performance of model.
    """

    # Transform y zu y_log
    y_log = np.log(y)
    mean_log = np.median(y_log)
    y_log = np.where(y_log == np.inf, mean_log, y_log)
    y_log = np.where(y_log == -np.inf, mean_log, y_log)

    pipe = Pipeline([('scaler', MinMaxScaler()),
                     ('rfecv', RFECV(LinearRegression(), step=1, cv=5)),
                     ('regressor', LinearRegression())])

    pipe.fit(X, y_log)

    # Select features
    selected_features = X.columns[pipe.named_steps['rfecv'].support_]
    features_dict[(f"{uti}_exp_regression_rfecv")] = dict()
    features_dict[(f"{uti}_exp_regression_rfecv")][pt] = selected_features.tolist()

    # Train Model on selected Features
    pipe_model = Pipeline([('scaler', MinMaxScaler()), ('reg', LinearRegression())])
    X_selected = X[selected_features]
    pipe_model.fit(X_selected, y_log)

    y_pred_log = pipe_model.predict(X_selected)
    y_pred = np.exp(y_pred_log)

    # Save model
    model_name = (f"{uti}_{pt}_exp_regression_rfecv.sav")

    # Track performance
    model = 'exp_regression'
    feature = 'rfecv'
    model_performance(model, pt, feature, selected_features, model_name, X_selected, y, y_pred, y_log, y_pred_log)


# Exp bay Regression
def feature_exp_bay_regression(X, y, pt, search_direction):
    """
    :param pt: Product type
    :param X: X train values.
    :param y: y train values.
    :param search_direction: forward vs backward for search direction
    :return: Performance of model.
    """

    # Transform y zu y_log
    y_log = np.log(y)
    mean_log = np.median(y_log)
    y_log = np.where(y_log == np.inf, mean_log, y_log)
    y_log = np.where(y_log == -np.inf, mean_log, y_log)

    pipe = Pipeline([('scaler', MinMaxScaler()),
                     ('sfs', SFS(BayesianRidge(compute_score=True), direction=search_direction,
                                 scoring='neg_mean_squared_error', cv=5)),
                     ('regressor', BayesianRidge(compute_score=True))])

    pipe.fit(X, y_log)

    # Select features
    selected_features = X.columns[pipe.named_steps['sfs'].support_]
    features_dict[(f"{uti}_exp_bay_regression_{search_direction}")] = dict()
    features_dict[(f"{uti}_exp_bay_regression_{search_direction}")][pt] = selected_features.tolist()

    # Train Model on selected Features
    pipe_model = Pipeline([('scaler', MinMaxScaler()), ('reg', BayesianRidge(compute_score=True))])
    X_selected = X[selected_features]
    pipe_model.fit(X_selected, y_log)

    y_pred_log = pipe_model.predict(X_selected)
    y_pred = np.exp(y_pred_log)

    # Save model
    model_name = (f"{uti}_{pt}_exp_bay_regression_{search_direction}.sav")

    # Track performance
    model = 'exp_bay_regression'
    feature = search_direction
    model_performance(model, pt, feature, selected_features, model_name, X_selected, y, y_pred, y_log, y_pred_log)


# Exp Regression using Bay Regression and RFECV

def rfecv_feature_exp_bay_regression(X, y, pt):
    """
    :param pt: Product type
    :param X: X train values.
    :param y: y train values.
    :return: Performance of model.
    """

    # Transform y zu y_log
    y_log = np.log(y)
    mean_log = np.median(y_log)
    y_log = np.where(y_log == np.inf, mean_log, y_log)
    y_log = np.where(y_log == -np.inf, mean_log, y_log)

    pipe = Pipeline([('scaler', MinMaxScaler()),
                     ('rfecv', RFECV(BayesianRidge(compute_score=True), step=1, cv=5)),
                     ('regressor', BayesianRidge(compute_score=True))])

    pipe.fit(X, y_log)

    # Select features
    selected_features = X.columns[pipe.named_steps['rfecv'].support_]
    features_dict[(f"{uti}_exp_bay_regression_rfecv")] = dict()
    features_dict[(f"{uti}_exp_bay_regression_rfecv")][pt] = selected_features.tolist()

    # Train Model on selected Features
    pipe_model = Pipeline([('scaler', MinMaxScaler()), ('reg', BayesianRidge(compute_score=True))])
    X_selected = X[selected_features]
    pipe_model.fit(X_selected, y_log)

    y_pred_log = pipe_model.predict(X_selected)
    y_pred = np.exp(y_pred_log)

    # Save model
    model_name = (f"{uti}_{pt}_exp_bay_regression_rfecv.sav")

    # Track performance
    model = 'exp_bay_regression'
    feature = 'rfecv'
    model_performance(model, pt, feature, selected_features, model_name, X_selected, y, y_pred, y_log, y_pred_log)


# Run for all PT
for pt in df_train['product_type'].unique():
    print(pt)
    print(f"{pt} wird jetzt bearbeitet")
    df_pt_train = df_train.loc[df_train['product_type'] == pt]
    X_train = df_pt_train.drop(
        columns=['order_id', 'product_type', 'replication', 'demand', 'Unnamed: 0', 'Unnamed: 0.1' 'sftt'])
    y_train = df_pt_train['sftt']


    # RFECV
    rfecv_feature_lin_regression(X_train, y_train, pt)
    rfecv_feature_exp_regression(X_train, y_train, pt)
    rfecv_feature_exp_bay_regression(X_train, y_train, pt)

    # Forward
    search_direction = 'forward'
    feature_lin_regression(X_train, y_train, pt, search_direction)
    feature_exp_regression(X_train, y_train, pt, search_direction)
    feature_exp_bay_regression(X_train, y_train, pt, search_direction)

    # Backward
    search_direction = 'backward'
    feature_lin_regression(X_train, y_train, pt, search_direction)
    feature_exp_regression(X_train, y_train, pt, search_direction)
    feature_exp_bay_regression(X_train, y_train, pt, search_direction)
