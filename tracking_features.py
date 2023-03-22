import pandas as pd
import numpy as np
import statistics
import GlobaleVariables
from scipy import stats
import uuid
import db_export


def get_wip():
    """
    Calculates the number of work in progress
    :return: returns th wip
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        wip = 0
        for station in GlobaleVariables.stations_list:
            wip += len(station.machine.queue)
        wip += 3
        return wip


def expected_mean_processing_time(product_type):
    """
    Calculates the expected mean processing time for each product type. This depends on the number of stations that need to
    visited.
    :param product_type: The orders product
    :return: The expected mean processing time for the specific product type.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        exp_mean_processing_time = len(GlobaleVariables.routing.get(product_type)) * 90

        return exp_mean_processing_time


def number_order_queue_routing(product_type):
    """
    Calculates the number of waiting orders ahead a station on the orders route (depending on product type).
    :param product_type: The orders' product type.
    :return: The number of waiting orders ahead the stations on the routing.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        nb_queue_routing = 0

        for station_nr in GlobaleVariables.routing.get(product_type):
            nb_queue_routing += len(GlobaleVariables.stations_list[station_nr - 1].machine.queue)

        return nb_queue_routing


def queue_time_all_orders():
    """
    Calculates for all product types the MEAN, the MEDIAN, the MODE queue time.
    :return: TOTAL queue time, MEAN, MEDIAN and MODE.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        try:
            queue_time = GlobaleVariables.tracking_df['queue_time']

            mean_qt = np.mean(queue_time)
            median_qt = np.median(queue_time)
            mode_qt = stats.mode(queue_time)[0].item()

        except:
            mean_qt = 0
            median_qt = 0
            mode_qt = 0

        return mean_qt, median_qt, mode_qt


def queue_time_product_type(product_type):
    """
    Calculates for the same product type the  MEAN, MEDIAN and MODE queu time.
    :param product_type: Product Type of the order
    :return: MEAN, MEDIAN and MODE of the same product type.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        try:
            tracking_df = GlobaleVariables.tracking_df
            queue_time = tracking_df['queue_time'].loc[tracking_df['product_type'] == product_type]

            mean_qt = np.mean(queue_time)
            median_qt = np.median(queue_time)
            mode_qt = stats.mode(queue_time)[0].item()

        except:
            mean_qt = 0
            median_qt = 0
            mode_qt = 0

        return mean_qt, median_qt, mode_qt


def queue_time_station():
    """
    Calculates the MEAN, MEDIAN and MODE for each queue time of each station.
    :return: MEAN, MEDIAN and MODE of each station.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        try:
            mean_stations = []
            median_stations = []
            mode_stations = []
            for key in range(1, 4):
                queue_time = GlobaleVariables.queue_time_station_dict[key]

                mean_stations.append(np.mean(queue_time))
                median_stations.append(np.median(queue_time))
                mode_stations.append(stats.mode(queue_time)[0].item())

            # Mean values per station
            mean_station_1 = mean_stations[0]
            mean_station_2 = mean_stations[1]
            mean_station_3 = mean_stations[2]

            # Median values per station
            median_station_1 = median_stations[0]
            median_station_2 = median_stations[1]
            median_station_3 = median_stations[2]

            # Mode values per station
            mode_station_1 = mode_stations[0]
            mode_station_2 = mode_stations[1]
            mode_station_3 = mode_stations[2]

        except:
            mean_station_1 = mean_station_2 = mean_station_3 = median_station_1 = median_station_2 = median_station_3 = \
                mode_station_1 = mode_station_2 = mode_station_3 = 0

        return mean_station_1, mean_station_2, mean_station_3, median_station_1, median_station_2, median_station_3, \
               mode_station_1, mode_station_2, mode_station_3


def exponential_smoothing(data, alpha):
    """
    The exponential smoothing is calculated.
    :param data: Data, that should be smoothed.
    :param alpha: The alpha value for the smoothing.
    :return: The smoothed values.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        data = data.values
        smoothed_values = []
        smoothed = data[0]
        for value in data[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
            smoothed_values.append(smoothed)
        return smoothed_values


def queue_time_exp_smoothed():
    """
    Calculates first for all queue times exponentially smoothed values using the exponential smoothing function. First
    for alpha = 0.3, then for alpha = 0.15. Afterwards, for both the MEAN, MEDIAN and MODE are calculated.
    :return: The MEAN, MEDIAN, MODE for both alpha 0.3, and alpha 0.15
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        try:
            tracking_df = GlobaleVariables.tracking_df
            queue_time = tracking_df['queue_time']

            # Alpha = 0.3
            alpha = 0.3
            smoothed_queue_time = exponential_smoothing(queue_time, alpha)
            mean_smoothed_03 = np.mean(smoothed_queue_time)
            median_smoothed_03 = np.median(smoothed_queue_time)
            mode_smoothed_03 = stats.mode(smoothed_queue_time)[0].item()

            # Alpha = 0.15
            alpha = 0.15
            smoothed_queue_time = exponential_smoothing(queue_time, alpha)
            mean_smoothed_015 = np.mean(smoothed_queue_time)
            median_smoothed_015 = np.median(smoothed_queue_time)
            mode_smoothed_015 = stats.mode(smoothed_queue_time)[0].item()
        except:
            mean_smoothed_03 = median_smoothed_03 = mode_smoothed_03 = mean_smoothed_015 = median_smoothed_015 = \
                mode_smoothed_015 = 0

        return mean_smoothed_03, median_smoothed_03, mode_smoothed_03, mean_smoothed_015, median_smoothed_015, mode_smoothed_015


def queue_time_exp_smoothed_pt(product_type):
    """
        Calculates first for the queue times of the same product type exponentially smoothed values using the
        exponential smoothing function. First for alpha = 0.3, then for alpha = 0.15. Afterwards, for both the MEAN,
        MEDIAN and MODE are calculated.
        :param product_type: The orders prodcut type.
        :return: The MEAN, MEDIAN, MODE for both alpha 0.3, and alpha 0.15.
        """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        try:
            tracking_df = GlobaleVariables.tracking_df
            queue_time = tracking_df['queue_time'].loc[tracking_df['product_type'] == product_type]

            # Alpha = 0.3
            alpha = 0.3
            smoothed_queue_time = exponential_smoothing(queue_time, alpha)
            mean_smoothed_03 = np.mean(smoothed_queue_time)
            median_smoothed_03 = np.median(smoothed_queue_time)
            mode_smoothed_03 = stats.mode(smoothed_queue_time)[0].item()

            # Alpha = 0.15
            alpha = 0.15
            smoothed_queue_time = exponential_smoothing(queue_time, alpha)
            mean_smoothed_015 = np.mean(smoothed_queue_time)
            median_smoothed_015 = np.median(smoothed_queue_time)
            mode_smoothed_015 = stats.mode(smoothed_queue_time)[0].item()
        except:
            mean_smoothed_03 = median_smoothed_03 = mode_smoothed_03 = mean_smoothed_015 = median_smoothed_015 = \
                mode_smoothed_015 = 0

        return mean_smoothed_03, median_smoothed_03, mode_smoothed_03, mean_smoothed_015, median_smoothed_015, mode_smoothed_015


def queue_time_standard_dev():
    """
    Calculates for the queue_time of all product types the standard deviation. As the packages statistics requires a minimum
    number of input values the try exception clause is applied.
    :return: the queue_time standard deviation value of all queue_times.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        try:
            queue_time = GlobaleVariables.tracking_df['queue_time']
            queue_time_std = statistics.stdev(queue_time)
        except:
            queue_time_std = 0

        return queue_time_std


def queue_time_standard_dev_pt(product_type):
    """
    Calculates for the queue_time of the same product types the standard deviation. As the packages statistics requires
    a minimum number of input values the try exception clause is applied.
    :param product_type: The orders product type.
    :return: The queue_time standard deviation value of the same product types.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        tracking_df = GlobaleVariables.tracking_df
        try:
            queue_time = tracking_df['queue_time'].loc[tracking_df['product_type'] == product_type]
            queue_time_std = statistics.stdev(queue_time)
        except:
            queue_time_std = 0

        return queue_time_std


def last_sftt_1(product_type):
    """
    Returns the sftt for the lastest finished order of the same product type as the order.
    :param product_type: Product Type of the order.
    :return: The SFTT of the latest finished order from the same prodcut type.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        try:
            tracking_df = GlobaleVariables.tracking_df
            last_sftt = tracking_df['sftt'].loc[tracking_df['product_type'] == product_type].tail(1).item()
        except:
            last_sftt = 0

        return last_sftt


def last_sftt_10(product_type):
    """
    Returns the sftt for the 10 lastest finished order of the same product type as the order.
    :param product_type: Product Type of the order.
    :return: MEAN, MEDIAN and MODE of the 10 latest finished orders' SFTT.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        try:
            tracking_df = GlobaleVariables.tracking_df
            last_sftt = tracking_df['sftt'].loc[tracking_df['product_type'] == product_type].tail(10)

            mean_sftt_10 = np.mean(last_sftt)
            median_sftt_10 = np.median(last_sftt)
            mode_sftt_10 = stats.mode(last_sftt)[0].item()
        except:
            mean_sftt_10 = median_sftt_10 = mode_sftt_10 = 0

        return mean_sftt_10, median_sftt_10, mode_sftt_10


def last_sftt_20(product_type):
    """
    Returns the sftt for the 20 lastest finished order of the same product type as the order.
    :param product_type: Product Type of the order.
    :return: MEAN, MEDIAN and MODE of the 20 latest finished orders' SFTT.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        try:
            tracking_df = GlobaleVariables.tracking_df
            last_sftt = tracking_df['sftt'].loc[tracking_df['product_type'] == product_type].tail(20)

            mean_sftt_20 = np.mean(last_sftt)
            median_sftt_20 = np.median(last_sftt)
            mode_sftt_20 = stats.mode(last_sftt)[0].item()
        except:
            mean_sftt_20 = median_sftt_20 = mode_sftt_20 = 0

        return mean_sftt_20, median_sftt_20, mode_sftt_20


def last_sftt_30(product_type):
    """
    Returns the sftt for the 30 lastest finished order of the same product type as the order.
    :param product_type: Product Type of the order.
    :return: MEAN, MEDIAN and MODE of the 30 latest finished orders' SFTT.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        try:
            tracking_df = GlobaleVariables.tracking_df
            last_sftt = tracking_df['sftt'].loc[tracking_df['product_type'] == product_type].tail(30)

            mean_sftt_30 = np.mean(last_sftt)
            median_sftt_30 = np.median(last_sftt)
            mode_sftt_30 = stats.mode(last_sftt)[0].item()
        except:
            mean_sftt_30 = median_sftt_30 = mode_sftt_30 = 0

        return mean_sftt_30, median_sftt_30, mode_sftt_30


def last_sftt_40(product_type):
    """
    Returns the sftt for the 30 lastest finished order of the same product type as the order.
    :param product_type: Product Type of the order.
    :return: MEAN, MEDIAN and MODE of the 30 latest finished orders' SFTT.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        try:
            tracking_df = GlobaleVariables.tracking_df
            last_sftt = tracking_df['sftt'].loc[tracking_df['product_type'] == product_type].tail(30)

            mean_sftt_40 = np.mean(last_sftt)
            median_sftt_40 = np.median(last_sftt)
            mode_sftt_40 = stats.mode(last_sftt)[0].item()
        except:
            mean_sftt_40 = median_sftt_40 = mode_sftt_40 = 0

        return mean_sftt_40, median_sftt_40, mode_sftt_40


def last_sftt_50(product_type):
    """
    Returns the sftt for the 30 lastest finished order of the same product type as the order.
    :param product_type: Product Type of the order.
    :return: MEAN, MEDIAN and MODE of the 30 latest finished orders' SFTT.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        try:

            tracking_df = GlobaleVariables.tracking_df
            last_sftt = tracking_df['sftt'].loc[tracking_df['product_type'] == product_type].tail(30)

            mean_sftt_50 = np.mean(last_sftt)
            median_sftt_50 = np.median(last_sftt)
            mode_sftt_50 = stats.mode(last_sftt)[0].item()
        except:
            mean_sftt_50 = median_sftt_50 = mode_sftt_50 = 0

        return mean_sftt_50, median_sftt_50, mode_sftt_50


def sftt_exp_smoothed():
    """
    Calculate the MEAN, MEDIAN, MODE of the exponentially smoothed SFTTs' of all orders.
    :return: The MEAN, MEDIAN, MODE for both alpha 0.3, and alpha 0.15.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        try:
            sftt = GlobaleVariables.tracking_df['sftt']

            # Alpha = 0.3
            alpha = 0.3
            smoothed_sftt = exponential_smoothing(sftt, alpha)
            mean_smoothed_03 = np.mean(smoothed_sftt)
            median_smoothed_03 = np.median(smoothed_sftt)
            mode_smoothed_03 = stats.mode(smoothed_sftt)[0].item()

            # Alpha = 0.15
            alpha = 0.15
            smoothed_sftt = exponential_smoothing(sftt, alpha)
            mean_smoothed_015 = np.mean(smoothed_sftt)
            median_smoothed_015 = np.median(smoothed_sftt)
            mode_smoothed_015 = stats.mode(smoothed_sftt)[0].item()
        except:
            mean_smoothed_03 = median_smoothed_03 = mode_smoothed_03 = mean_smoothed_015 = \
                median_smoothed_015 = mode_smoothed_015 = 0

        return mean_smoothed_03, median_smoothed_03, mode_smoothed_03, mean_smoothed_015, median_smoothed_015, mode_smoothed_015


def sftt_exp_smoothed_pt(product_type):
    """
    Calculates first for the sftt of the same product type exponentially smoothed values using the
    exponential smoothing function. First for alpha = 0.3, then for alpha = 0.15. Afterwards, for both the MEAN, MEDIAN
    and MODE are calculated.
    :param product_type: The orders product type.
    :return: The MEAN, MEDIAN, MODE for both alpha 0.3, and alpha 0.15.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        try:
            tracking_df = GlobaleVariables.tracking_df
            sftt = tracking_df['sftt'].loc[tracking_df['product_type'] == product_type]

            # Alpha = 0.3
            alpha = 0.3
            smoothed_sftt = exponential_smoothing(sftt, alpha)
            mean_smoothed_03 = np.mean(smoothed_sftt)
            median_smoothed_03 = np.median(smoothed_sftt)
            mode_smoothed_03 = stats.mode(smoothed_sftt)[0].item()

            # Alpha = 0.15
            alpha = 0.15
            smoothed_sftt = exponential_smoothing(sftt, alpha)
            mean_smoothed_015 = np.mean(smoothed_sftt)
            median_smoothed_015 = np.median(smoothed_sftt)
            mode_smoothed_015 = stats.mode(smoothed_sftt)[0].item()
        except:
            mean_smoothed_03 = median_smoothed_03 = mode_smoothed_03 = mean_smoothed_015 = median_smoothed_015 = \
                mode_smoothed_015 = 0

        return mean_smoothed_03, median_smoothed_03, mode_smoothed_03, mean_smoothed_015, median_smoothed_015, mode_smoothed_015


def sftt_standard_dev():
    """
    Calculates for the sftt of all product types the standard deviation. As the packages statistics requires a minimum
    number of input values the try exception clause is applied.
    :return: the sftt standard deviation value of all sftt.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        try:
            sftt = GlobaleVariables.tracking_df['sftt']
            sftt_std = statistics.stdev(sftt)
        except:
            sftt_std = 0

        return sftt_std


def sftt_standard_dev_pt(product_type):
    """
    Calculates for the sftt of the same product types the standard deviation. As the packages statistics requires a
    minimum number of input values the try exception clause is applied.
    :param product_type: The orders product type.
    :return: the sftt standard deviation value of teh same product type.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        tracking_df = GlobaleVariables.tracking_df
        try:
            sftt = tracking_df['sftt'].loc[tracking_df['product_type'] == product_type]
            sftt_std = statistics.stdev(sftt)
        except:
            sftt_std = 0

        return sftt_std


def collect_features(order, product_type):
    """
    This function collects all feautres and then calls the deb export function
    :param order: The referring order.
    :param product_type: te referring product type.
    """
    if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
        order_id = order.order_id
        product_type = product_type
        wip = get_wip()
        expected_mean_p_t = expected_mean_processing_time(product_type)
        nb_orders_queue_routing = number_order_queue_routing(product_type)
        queue_time_all_orders_mean, queue_time_all_orders_median, queue_time_all_orders_mode = queue_time_all_orders()
        queue_time_pt_mean, queue_time_pt_median, queue_time_pt_mode = queue_time_product_type(product_type)
        queue_time_st1_mean, queue_time_st2_mean, queue_time_st3_mean, queue_time_st1_median, queue_time_st2_median, \
        queue_time_st3_median, queue_time_st1_mode, queue_time_st2_mode, queue_time_st3_mode = queue_time_station()
        queue_time_exp_smoothed_mean_03, queue_time_exp_smoothed_median_03, queue_time_exp_smoothed_mode_03, \
        queue_time_exp_smoothed_mean_015, queue_time_exp_smoothed_median_015, \
        queue_time_exp_smoothed_mode_015 = queue_time_exp_smoothed()
        queue_time_exp_smoothed_pt_mean_03, queue_time_exp_smoothed_pt_median_03, queue_time_exp_smoothed_pt_mode_03, \
        queue_time_exp_smoothed_pt_mean_015, queue_time_exp_smoothed_pt_median_015, \
        queue_time_exp_smoothed_pt_mode_015 = queue_time_exp_smoothed_pt(product_type)
        queue_time_std_dev = queue_time_standard_dev()
        queue_time_std_dv_pt = queue_time_standard_dev_pt(product_type)
        last_sftt = last_sftt_1(product_type)
        last_sftt_10_mean, last_sftt_10_median, last_sftt_10_mode = last_sftt_10(product_type)
        last_sftt_20_mean, last_sftt_20_median, last_sftt_20_mode = last_sftt_20(product_type)
        last_sftt_30_mean, last_sftt_30_median, last_sftt_30_mode = last_sftt_30(product_type)
        last_sftt_40_mean, last_sftt_40_median, last_sftt_40_mode = last_sftt_40(product_type)
        last_sftt_50_mean, last_sftt_50_median, last_sftt_50_mode = last_sftt_50(product_type)
        sftt_exp_smoothed_mean_03, sftt_exp_smoothed_median_03, sftt_exp_smoothed_mode_03, sftt_exp_smoothed_mean_015, \
        sftt_exp_smoothed_median_015, sftt_exp_smoothed_mode_015 = sftt_exp_smoothed()
        sftt_exp_smoothed_pt_mean_03, sftt_exp_smoothed_pt_median_03, sftt_exp_smoothed_pt_mode_03, \
        sftt_exp_smoothed_pt_mean_015, sftt_exp_smoothed_pt_median_015, \
        sftt_exp_smoothed_pt_mode_015 = sftt_exp_smoothed_pt(product_type)
        sftt_std_dev = sftt_standard_dev()
        sftt_std_dev_pt = sftt_standard_dev_pt(product_type)
        random_key = str(uuid.uuid4())

        new_df = pd.DataFrame({'order_id': order_id, 'product_type': product_type, 'wip': wip,
                               'expected_mean_p_t': expected_mean_p_t,
                               'nb_orders_queue_routing': nb_orders_queue_routing,
                               'queue_time_all_orders_mean': queue_time_all_orders_mean, 'queue_time_all_orders_median':
                                   queue_time_all_orders_median,
                               'queue_time_all_orders_mode': queue_time_all_orders_mode,
                               'queue_time_pt_mean': queue_time_pt_mean, 'queue_time_pt_median': queue_time_pt_median,
                               'queue_time_pt_mode': queue_time_pt_mode,
                               'queue_time_station1_mean': queue_time_st1_mean,
                               'queue_time_station1_median': queue_time_st1_median,
                               'queue_time_station1_mode': queue_time_st1_mode,
                               'queue_time_station2_mean': queue_time_st2_mean,
                               'queue_time_station2_median': queue_time_st2_median,
                               'queue_time_station2_mode': queue_time_st2_mode,
                               'queue_time_station3_mean': queue_time_st3_mean,
                               'queue_time_station3_median': queue_time_st3_median,
                               'queue_time_station3_mode': queue_time_st3_mode,
                               'queue_time_exp_smoothed_mean_03': queue_time_exp_smoothed_mean_03,
                               'queue_time_exp_smoothed_median_03': queue_time_exp_smoothed_median_03,
                               'queue_time_exp_smoothed_mode_03': queue_time_exp_smoothed_mode_03,
                               'queue_time_exp_smoothed_mean_015': queue_time_exp_smoothed_mean_015,
                               'queue_time_exp_smoothed_median_015': queue_time_exp_smoothed_median_015,
                               'queue_time_exp_smoothed_mode_015': queue_time_exp_smoothed_mode_015,
                               'queue_time_exp_smoothed_pt_mean_03': queue_time_exp_smoothed_pt_mean_03,
                               'queue_time_exp_smoothed_pt_median_03': queue_time_exp_smoothed_pt_median_03,
                               'queue_time_exp_smoothed_pt_mode_03': queue_time_exp_smoothed_pt_mode_03,
                               'queue_time_exp_smoothed_pt_mean_015': queue_time_exp_smoothed_pt_mean_015,
                               'queue_time_exp_smoothed_pt_median_015': queue_time_exp_smoothed_pt_median_015,
                               'queue_time_exp_smoothed_pt_mode_015': queue_time_exp_smoothed_pt_mode_015,
                               'queue_time_std_dv': queue_time_std_dev,
                               'queue_time_std_dv_pt': queue_time_std_dv_pt,
                               'last_sftt_1': last_sftt,
                               'last_sftt_10_mean': last_sftt_10_mean,
                               'last_sftt_10_median': last_sftt_10_median,
                               'last_sftt_10_mode': last_sftt_10_mode,
                               'last_sftt_20_mean': last_sftt_20_mean,
                               'last_sftt_20_median': last_sftt_20_median,
                               'last_sftt_20_mode': last_sftt_20_mode,
                               'last_sftt_30_mean': last_sftt_30_mean,
                               'last_sftt_30_median': last_sftt_30_median,
                               'last_sftt_30_mode': last_sftt_30_mode,
                               'last_sftt_40_mean': last_sftt_40_mean,
                               'last_sftt_40_median': last_sftt_40_median,
                               'last_sftt_40_mode': last_sftt_40_mode,
                               'last_sftt_50_mean': last_sftt_50_mean,
                               'last_sftt_50_median': last_sftt_50_median,
                               'last_sftt_50_mode': last_sftt_50_mode,
                               'sftt_exp_smoothed_mean_03': sftt_exp_smoothed_mean_03,
                               'sftt_exp_smoothed_median_03': sftt_exp_smoothed_median_03,
                               'sftt_exp_smoothed_mode_03': sftt_exp_smoothed_mode_03,
                               'sftt_exp_smoothed_mean_015': sftt_exp_smoothed_mean_015,
                               'sftt_exp_smoothed_median_015': sftt_exp_smoothed_median_015,
                               'sftt_exp_smoothed_mode_015': sftt_exp_smoothed_mode_015,
                               'sftt_exp_smoothed_pt_mean_03': sftt_exp_smoothed_pt_mean_03,
                               'sftt_exp_smoothed_pt_median_03': sftt_exp_smoothed_pt_median_03,
                               'sftt_exp_smoothed_pt_mode_03': sftt_exp_smoothed_pt_mode_03,
                               'sftt_exp_smoothed_pt_mean_015': sftt_exp_smoothed_pt_mean_015,
                               'sftt_exp_smoothed_pt_median_015': sftt_exp_smoothed_pt_median_015,
                               'sftt_exp_smoothed_pt_mode_015': sftt_exp_smoothed_pt_mode_015,
                               'sftt_std_dev': sftt_std_dev,
                               'sftt_std_dev_pt': sftt_std_dev_pt
                               }, index=[order_id])

        GlobaleVariables.features_df = pd.concat([GlobaleVariables.features_df, new_df])

        db_export.export_to_db_features(order_id, product_type, GlobaleVariables.SIM_ROUND, GlobaleVariables.SCENARIO,
                                        wip, expected_mean_p_t, nb_orders_queue_routing, queue_time_all_orders_mean,
                                        queue_time_all_orders_median, queue_time_all_orders_mode, queue_time_pt_mean,
                                        queue_time_pt_median, queue_time_pt_mode, queue_time_st1_mean,
                                        queue_time_st1_median,
                                        queue_time_st1_mode, queue_time_st2_mean, queue_time_st2_median,
                                        queue_time_st2_mode,
                                        queue_time_st3_mean, queue_time_st3_median, queue_time_st3_mode,
                                        queue_time_exp_smoothed_mean_03, queue_time_exp_smoothed_median_03,
                                        queue_time_exp_smoothed_mode_03, queue_time_exp_smoothed_mean_015,
                                        queue_time_exp_smoothed_median_015, queue_time_exp_smoothed_mode_015,
                                        queue_time_exp_smoothed_pt_mean_03, queue_time_exp_smoothed_pt_median_03,
                                        queue_time_exp_smoothed_pt_mode_03, queue_time_exp_smoothed_pt_mean_015,
                                        queue_time_exp_smoothed_pt_median_015, queue_time_exp_smoothed_pt_mode_015,
                                        queue_time_std_dev, queue_time_std_dv_pt, last_sftt, last_sftt_10_mean,
                                        last_sftt_10_median, last_sftt_10_mode, last_sftt_20_mean, last_sftt_20_median,
                                        last_sftt_20_mode, last_sftt_30_mean, last_sftt_30_median, last_sftt_30_mode,
                                        last_sftt_40_mean, last_sftt_40_median, last_sftt_40_mode, last_sftt_50_mean,
                                        last_sftt_50_median, last_sftt_50_mode, sftt_exp_smoothed_mean_03,
                                        sftt_exp_smoothed_median_03, sftt_exp_smoothed_mode_03,
                                        sftt_exp_smoothed_mean_015,
                                        sftt_exp_smoothed_median_015, sftt_exp_smoothed_mode_015,
                                        sftt_exp_smoothed_pt_mean_03, sftt_exp_smoothed_pt_median_03,
                                        sftt_exp_smoothed_pt_mode_03, sftt_exp_smoothed_pt_mean_015,
                                        sftt_exp_smoothed_pt_median_015, sftt_exp_smoothed_pt_mode_015, sftt_std_dev,
                                        sftt_std_dev_pt, random_key)
