import GlobaleVariables
import pandas as pd
import db_export
import uuid
import validation as v


def order_track(order, env):
    """
    Tracks the order_id, product_type, due_date, created_time and created_period for each order.
    :param order: Order that was created.
    :param env: Simpy Environment.
    :return: Stores the attributes in GlobaleVariables.order_tracking dict.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        GlobaleVariables.order_tracking[order.order_id] = dict()
        GlobaleVariables.order_tracking[order.order_id]['product_type'] = order.product_type
        GlobaleVariables.order_tracking[order.order_id]['due_date'] = order.due_date
        GlobaleVariables.order_tracking[order.order_id]['time_created'] = env.now
        GlobaleVariables.order_tracking[order.order_id]['period_created'] = GlobaleVariables.period


def order_track_release(order, env):
    """
    Tracks the release of each order.
    :param order: Order that was release.
    :param env: Simpy Environment.
    :return: Stores the release time in GlobaleVariables.order_tracking dict.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        if order.order_id in GlobaleVariables.order_tracking.keys():
            GlobaleVariables.order_tracking[order.order_id]['time_release'] = env.now


def order_track_request(order_id, env):
    """
    Tracks the request time of the order.
    :param order_id: Order that request the station.
    :param env: Simpy Environment.
    :return: Stores the attributes in GlobaleVariables.order_tracking dict.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        if order_id in GlobaleVariables.order_tracking.keys():
            GlobaleVariables.order_tracking[order_id]['request'] = env.now


def stations_queue_tracking_order(station, queue_time):
    """
    Appends the queue time of an order to the referring station.
    :param queue_time: The
    :return: Appends the queue time to the stations queue length
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        if station.number not in GlobaleVariables.queue_time_station_dict.keys():
            GlobaleVariables.queue_time_station_dict[station.number] = [queue_time]
        else:
            GlobaleVariables.queue_time_station_dict[station.number].append(queue_time)


def order_track_queue_time(order_id, env, station):
    """
    Tracks the queue time of each order and stores the value in the GlobaleVariables.order_tracking dict.
    :param order_id: Order_id of the order.
    :param env: Simpy Environment.
    :return: Updated GlobaleVariables.order_tracking dict.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        if order_id in GlobaleVariables.order_tracking.keys():
            queue_time = env.now - GlobaleVariables.order_tracking[order_id]['request']
            if 'queue_time' not in GlobaleVariables.order_tracking[order_id].keys():
                GlobaleVariables.order_tracking[order_id]['queue_time'] = queue_time
            else:
                GlobaleVariables.order_tracking[order_id]['queue_time'] = GlobaleVariables.order_tracking[order_id][
                                                                              'queue_time'] + queue_time

            # Call stations_queue_tracking_order to store the queue_time in the dict
            # queue_time = env.now - GlobaleVariables.order_tracking[order_id]['request']
            stations_queue_tracking_order(station, queue_time)


def order_tracking_to_df(order_id):
    """
    Stores the finished order attributes in the GlobalVariables.order_tracking dataframe. Then deletes the order from
    the dictionary.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        if order_id in GlobaleVariables.order_tracking.keys():
            new_dict = GlobaleVariables.order_tracking.pop(order_id)
            new_dict['order_id'] = order_id
            new_df = pd.DataFrame(new_dict, index=[order_id])
            new_df.index.names = ['order_id']
            GlobaleVariables.tracking_df = pd.concat([GlobaleVariables.tracking_df, new_df])


def order_track_sftt(order_id, product_type, env, station, due_date):
    """
    Calculates the SFTT for each order. Then calls the function order-tracking_to_df
    :param product_type: Product Type of the order.
    :param order_id: Order ID.
    :param station: Current station.
    :param env: Simpy Environment.
    :return: Stores the SFTT in GlobaleVariables.order_tracking dict.
    """
    # Check whether last station
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        if order_id in GlobaleVariables.order_tracking.keys():
            if station.number == GlobaleVariables.routing.get(product_type)[-1]:
                release_time = GlobaleVariables.order_tracking[order_id]['time_release']
                GlobaleVariables.order_tracking[order_id]['time_finished'] = env.now
                GlobaleVariables.order_tracking[order_id]['period_finished'] = GlobaleVariables.period
                GlobaleVariables.order_tracking[order_id]['sftt'] = env.now - release_time

                random_key = str(uuid.uuid4())

                # Export to DB
                db_export.export_to_db_finished_orders(order_id, product_type, GlobaleVariables.SIM_ROUND,
                                                       GlobaleVariables.SCENARIO, due_date,
                                                       GlobaleVariables.order_tracking[order_id]['created_time'],
                                                       GlobaleVariables.order_tracking[order_id]['created_period'],
                                                       GlobaleVariables.order_tracking[order_id]['time_release'],
                                                       GlobaleVariables.order_tracking[order_id]['time_finished'],
                                                       GlobaleVariables.order_tracking[order_id]['period_finished'],
                                                       GlobaleVariables.order_tracking[order_id]['sftt'],
                                                       random_key)

                # Target Tracking
                new_df = pd.DataFrame(
                    {'order_id': order_id, 'product_type': product_type, 'replication': GlobaleVariables.SIM_ROUND,
                     'demand': GlobaleVariables.SCENARIO, 'dude_date': due_date,
                     'time_created': GlobaleVariables.order_tracking[order_id]['created_time'],
                     'period_created': GlobaleVariables.order_tracking[order_id]['created_period'],
                     'time_released': GlobaleVariables.order_tracking[order_id]['time_release'],
                     'time_finished': GlobaleVariables.order_tracking[order_id]['time_finished'],
                     'period_finished': GlobaleVariables.order_tracking[order_id]['period_finished'],
                     'sftt': GlobaleVariables.order_tracking[order_id]['sftt']
                     }, index=[order_id])

                GlobaleVariables.target_df = pd.concat([GlobaleVariables.target_df, new_df])

                # Track validation
                v.track_validation(order_id)

                # Call function order_tracking_to_df
                order_tracking_to_df(order_id)

                # Increase leaves shop
                GlobaleVariables.leaves_shop += 1


def track_utilization(station, processing_time, env):
    """
    This function append the processing time of each station and tracks the utilization.
    :param station: The station the order is processed on.
    :param processing_time: The processing time the order needed.
    :return:
    """
    if station.number in GlobaleVariables.utilization_dict.keys():
        GlobaleVariables.utilization_dict[station.number].append(processing_time)
    else:
        GlobaleVariables.utilization_dict[station.number] = [processing_time]
