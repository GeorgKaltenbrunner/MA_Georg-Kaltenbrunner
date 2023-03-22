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