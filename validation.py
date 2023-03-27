import GlobaleVariables
import numpy as np
import pandas as pd


def increase_finished_after_einschwing():
    """
    This function tracks the number of finished orders after the EINSCHWINGPHASE
    """
    # Increase
    GlobaleVariables.finished_after_einschwing += 1


def count_early_tardy_orders(order_id):
    """
    This functionsntracks the number of early/tardy orders and the earliness/tardiness.
    :param order_id: The finished orders id.
    """

    if GlobaleVariables.order_tracking[order_id]['time_finished'] < GlobaleVariables.order_tracking[order_id][
        'due_date']:
        GlobaleVariables.early_orders += 1
        earliness = GlobaleVariables.order_tracking[order_id]['due_date'] - GlobaleVariables.order_tracking[order_id][
            'time_finished']
        GlobaleVariables.earliness.append(earliness)
    else:
        GlobaleVariables.tardy_orders += 1
        tardiness = GlobaleVariables.order_tracking[order_id]['time_finished'] - \
                    GlobaleVariables.order_tracking[order_id][
                        'due_date']
        GlobaleVariables.tardiness.append(tardiness)


def average_sftt(order_id):
    """
    This function tracks the sftt of the finished order.
    :param order_id: The finished orders id.
    """
    GlobaleVariables.sftt_track.append(GlobaleVariables.order_tracking[order_id]['sftt'])


def track_validation(order_id):
    """
    This function calls the different functions to track the evaluation.
    :param order_id: The finished orders id
    """
    increase_finished_after_einschwing()
    count_early_tardy_orders(order_id)
    average_sftt(order_id)
    
def final_validation():
    """
    This function puts the validation information together and exports it as validation excel.
    :return: An Excel is created and exported.
    """
    early_orders = GlobaleVariables.early_orders
    tardy_orders = GlobaleVariables.tardy_orders
    early_ratio = GlobaleVariables.early_orders / GlobaleVariables.finished_after_einschwing
    tardy_ratio = tardy_orders / GlobaleVariables.finished_after_einschwing
    avg_sftt = np.array(GlobaleVariables.sftt_track).mean()
    utilization_station1 = sum(GlobaleVariables.utilization_dict[1]) / GlobaleVariables.SIM_TIME
    utilization_station2 = sum(GlobaleVariables.utilization_dict[2]) / GlobaleVariables.SIM_TIME
    utilization_station3 = sum(GlobaleVariables.utilization_dict[3]) / GlobaleVariables.SIM_TIME

    validation_df = pd.DataFrame({'early_orders': early_orders,
                                  'tardy_orders': tardy_orders,
                                  'early_ratio': early_ratio,
                                  'tardy_ratio': tardy_ratio,
                                  'mean_earliness': np.mean(
                                      GlobaleVariables.earliness),
                                  'mean_tardiness': np.mean(
                                      GlobaleVariables.tardiness),
                                  'avg_sftt': avg_sftt,
                                  'util_station_1': utilization_station1,
                                  'util_station_2': utilization_station2,
                                  'util_station_3': utilization_station3},
                                 index=[GlobaleVariables.SIM_ROUND])
