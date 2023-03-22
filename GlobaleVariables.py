import pandas as pd

# ---------------------------------------------------------------------------------------------------------------------
# Simulation Initials
EINLAUF_PERIODEN = 150
PERIOD_LENGTH = 1440
NUMBER_PERIODS = 1820
SIM_TIME = (EINLAUF_PERIODEN + NUMBER_PERIODS) * PERIOD_LENGTH
NEW_ORDERS_TIME = 68
SIM_ROUND = int
SCENARIO = 95

# 'avg_sftt', 'last_sftt', 'linear_regression', 'exp_regression', 'exp_bay_regression'
model = 'linear_regression'

# ---------------------------------------------------------------------------------------------------------------------
# Order and period IDs
# For orders
order_id = 1
period = 1

# ---------------------------------------------------------------------------------------------------------------------
# Routing
routing = {1: [1, 2, 3], 2: [2, 1, 3], 3: [3, 2, 1], 4: [1, 3, 2], 5: [2, 3, 1], 6: [3, 1, 2],
           7: [1, 2],
           8: [2, 1], 9: [3, 2], 10: [1, 3], 11: [2, 3], 12: [3, 1], 13: [1], 14: [2], 15: [3]}

stations_list = []
order_pool = []
order_pool_dict = dict()

# ---------------------------------------------------------------------------------------------------------------------
# Tracking
order_tracking = dict()
tracking_df = pd.DataFrame()
features_df = pd.DataFrame()
target_df = pd.DataFrame()

# ---------------------------------------------------------------------------------------------------------------------
# WIP
enters_shop = 0
leaves_shop = 0

# ---------------------------------------------------------------------------------------------------------------------
# Queuing
queue_time_station_dict = dict()

# ---------------------------------------------------------------------------------------------------------------------
# Utilization
utilization_dict = dict()

# ---------------------------------------------------------------------------------------------------------------------
# Validation
finished_after_einschwing = 0
created_after_einschwing = 0
early_orders = 0
tardy_orders = 0
sftt_track = []
earliness = []
tardiness = []
processing_time_list = []

# ---------------------------------------------------------------------------------------------------------------------
# Read best models
best_models = pd.DataFrame()
