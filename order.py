import random
import GlobaleVariables
import numpy as np
from scipy.stats import expon
import tracking_order as tk
import processing_time as pro_time
import tracking_features as tk_f
import release
import prediction

# Random seed
np.random.seed(seed=42)


class Order:
    """
    In this class first orders are generated. Afterwards sent to the stations on the routing where the order
    is handled. Also, the tracking for each order is done here.
    """

    def __init__(self, env, order_id, product_type, due_date, period):
        """
        Here the variables for the orders are defined.
        :param env: SimPy Environment()
        """
        self.env = env
        self.order_id = order_id  # Identifier for each order
        self.product_type = product_type
        self.due_date = due_date
        self.period = period  # Period the order is CREATED
        self.demand_time = int  # Time after which a new order is created
        self.demand_start = 0
        self.request = int  # -> when the order gets at the station
        self.prd = 0
        self.prd_period = 0

    def handle_order(self, station):
        """
        The order request the station. Whether the order needs to wait or is immediately processed.
        :param station: The station for the order on the routing.
        """

        # Make request
        with station.machine.request() as request:
            # Track request
            tk.order_track_request(self.order_id, self.env)

            yield request

            # Get Processing time with upper limit 360
            processing_time = pro_time.proc_time()
            # Track queue time of order
            tk.order_track_queue_time(self.order_id, self.env, station)
            # Track utilization
            tk.track_utilization(station, processing_time, self.env)

            yield self.env.timeout(processing_time)

            tk.order_track_sftt(self.order_id, self.product_type, self.env, station, self.due_date)

    def get_station(self):
        """
        The next station on the product types routing is selected.
        :return: Sending the order to the next station (handle_order() ).
        """

        stations = GlobaleVariables.routing.get(self.product_type)

        # Iterate over each station
        for station in stations:
            yield self.env.process(self.handle_order(GlobaleVariables.stations_list[station - 1]))

    def generate_orders(self):
        """
        New orders are created.
        :return: New order that is sent to get_station()
        """

        # Global order_id
        GlobaleVariables.order_id = 1

        # Order attributes
        self.order_id = GlobaleVariables.order_id
        self.product_type = random.randint(1, 15)
        self.due_date = self.env.now + (random.randint(2, 15) * GlobaleVariables.PERIOD_LENGTH)
        self.period = GlobaleVariables.period

        # New Order
        order = Order(self.env, self.order_id, self.product_type, self.due_date, self.period)

        # Track the attributes
        tk.order_track(order, self.env)
        tk_f.collect_features(order, self.product_type)

        # Prediction
        prediction.predict_sftt(order)

        # Append to order_pool_dict
        if order.prd_period in GlobaleVariables.order_pool_dict.keys():
            GlobaleVariables.order_pool_dict[order.prd_period].append(order)
        else:
            GlobaleVariables.order_pool_dict[order.prd_period] = [order]

        # Append order to order_pool
        GlobaleVariables.order_pool.append(order)

        # Define demand (time after which a new order is created)
        self.demand_time = expon.rvs(scale=GlobaleVariables.NEW_ORDERS_TIME).round()
        self.demand_start = 0
        while True:
            yield self.env.timeout(1)

            if self.env.now >= (self.demand_start + self.demand_time):

                # Increase order_id
                GlobaleVariables.order_id += 1
                if GlobaleVariables.period >= GlobaleVariables.EINLAUF_PERIODEN:
                    GlobaleVariables.created_after_einschwing += 1

                # Order attributes
                self.order_id = GlobaleVariables.order_id
                self.product_type = random.randint(1, 15)
                self.due_date = self.env.now + (random.randint(2, 15) * GlobaleVariables.PERIOD_LENGTH)
                self.period = GlobaleVariables.period

                # New order
                order = Order(self.env, self.order_id, self.product_type, self.due_date, self.period)

                # Track the attributes
                tk.order_track(order, self.env)
                tk_f.collect_features(order, self.product_type)

                # Predict SFTT
                prediction.predict_sftt(order)

                # Append to order dict
                if order.prd_period in GlobaleVariables.order_pool_dict.keys():
                    GlobaleVariables.order_pool_dict[order.prd_period].append(order)
                else:
                    GlobaleVariables.order_pool_dict[order.prd_period] = [order]

                # Append order to the order pool
                GlobaleVariables.order_pool.append(order)

                # Redefine Demand
                self.demand_time = expon.rvs(scale=GlobaleVariables.NEW_ORDERS_TIME).round()
                self.demand_start = self.env.now

            if self.env.now == (GlobaleVariables.period * GlobaleVariables.PERIOD_LENGTH):
                print(GlobaleVariables.period)
                # Insert Release and sorting approach
                for order in release.earliest_prd(release.PRD(GlobaleVariables.order_pool_dict)):
                    # Track release time
                    tk.order_track_release(order, self.env)

                    # Send order to the first stations
                    self.env.process(order.get_station())
                    GlobaleVariables.enters_shop += 1

                # Increase Period
                GlobaleVariables.period += 1
