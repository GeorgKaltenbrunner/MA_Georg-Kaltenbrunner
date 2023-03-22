import GlobaleVariables


def IR(order_pool):
    """
    This functions represents an Interval Release. So all order in the order pool are release to the shop floor.
    Deletes the previous order pool entries.
    :param order_pool: A list containing all the created orders.
    :return: A list with orders that are to be released.
    """
    release_list = []
    release_list.extend(order_pool)
    order_pool.clear()

    return release_list


def PRD(order_pool_dict):
    """
    This function filters the order pool dict, that all orders with the referring period are released.
    :param order_pool_dict: The exisitng dict where all created orders are stored in.
    :return: A list of orders to be released.
    """
    try:
        release_list = []
        for period in range(1, GlobaleVariables.period + 1):
            if period in order_pool_dict.keys():
                release_list.extend(order_pool_dict.get(period))
                del order_pool_dict[period]
    except:
        release_list = []

    return release_list


def earliest_prd(order_pool):
    """
    This function sorts the given order pool by the prd ascending PRD.
    :param order_pool: Order_pool oft the order to be released in this period.
    :return: Sorted list to be released first.
    """
    return sorted(order_pool, key=lambda x: x.prd)


def edd(order_pool):
    """
    This function sorts the given order pool by DD.
    :param order_pool: order_pool oft the order to be released in this period.
    :return: Sorted list to be released first.
    """
    return sorted(order_pool, key=lambda x: x.due_date)
