import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge, LinearRegression
import GlobaleVariables


def predict_sftt(order):
    """
    This function predicts the SFTT.
    :param order: The newly created order.
    :return: THe predicted SFTT.
    """
    # Filter GlobaleVariables on pt

    if GlobaleVariables.model == 'avg_sftt':
        try:
            tracking_df = GlobaleVariables.tracking_df
            y_pred = np.mean(tracking_df['sftt'].loc[tracking_df['product_type'] == order.product_type])

            # Calculate PRD
            order.prd = order.due_date - y_pred
            order.prd_period = int((order.due_date - y_pred) / 1440)
            if order.prd_period < GlobaleVariables.period:
                order.prd_period = GlobaleVariables.period

        except:
            order.prd_period = GlobaleVariables.period

    elif GlobaleVariables.model == 'last_sftt':
        try:
            tracking_df = GlobaleVariables.tracking_df
            y_pred = tracking_df['sftt'].loc[tracking_df['product_type'] == order.product_type].tail(1).item()

            # Calculate PRD
            order.prd = order.due_date - y_pred
            order.prd_period = int((order.due_date - y_pred) / 1440)
            if order.prd_period < GlobaleVariables.period:
                order.prd_period = GlobaleVariables.period

        except:
            order.prd_period = GlobaleVariables.period
    else:
        try:
            # Get the finished orders for feature selection
            filtered_finished = GlobaleVariables.tracking_df.loc[
                GlobaleVariables.tracking_df['product_type'] == order.product_type]
            features_df = GlobaleVariables.features_df
            filtered_df = filtered_finished.merge(features_df, how='left', left_on=filtered_finished["order_id"],
                                                  right_on=features_df["order_id"])

            # Get the features from the best model
            best_models = GlobaleVariables.best_models
            selected_features = eval(best_models.loc[best_models['product_type'] == order.product_type].features.item())

            # Get the features
            X_train = filtered_df[selected_features]
            y = filtered_df['sftt']

            # For new order features selection
            features_new_order = features_df.loc[features_df['order_id'] == order.order_id]
            features_new_order['time_created'] = GlobaleVariables.order_tracking[order.order_id]['time_created']
            features_new_order['period_created'] = GlobaleVariables.order_tracking[order.order_id]['period_created']
            X_prediction = features_new_order[selected_features]

            # Get model
            model_dict = {'linear_regression': LinearRegression(),
                          'exp_regression': LinearRegression(),
                          'exp_bay_regression': BayesianRidge()}

            model = model_dict.get(GlobaleVariables.model)

            # Check whether lin or exp
            if 'exp' in GlobaleVariables.model:
                try:
                    # Log sftt
                    y_log = np.log(y)
                    mean_log = np.median(y_log)
                    y_log = np.where(y_log == np.inf, mean_log, y_log)
                    y_log = np.where(y_log == -np.inf, mean_log, y_log)

                    # Create Pipeline
                    pipe = Pipeline([('scaler', MinMaxScaler()), ('reg', model)])

                    # Fit Pipeline
                    pipe.fit(X_train, y_log)

                    # Predict
                    y_pred = np.exp(pipe.predict(X_prediction)).item()

                    # Calculate PRD
                    order.prd = order.due_date - y_pred
                    order.prd_period = int((order.due_date - y_pred) / 1440)
                    if order.prd_period < GlobaleVariables.period:
                        order.prd_period = GlobaleVariables.period

                except:
                    order.prd_period = GlobaleVariables.period

            else:
                try:
                    # Create Pipeline
                    pipe = Pipeline([('scaler', MinMaxScaler()), ('reg', model)])

                    # Fit Pipeline
                    pipe.fit(X_train, y)

                    # Predict
                    y_pred = pipe.predict(X_prediction)

                    # Calculate PRD
                    order.prd = order.due_date - y_pred
                    order.prd_period = int((order.due_date - y_pred) / 1440)
                    if order.prd_period < GlobaleVariables.period:
                        order.prd_period = GlobaleVariables.period

                except:
                    order.prd_period = GlobaleVariables.period
        except Exception as e:
            e
            order.prd_period = GlobaleVariables.period
