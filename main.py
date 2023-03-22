import GlobaleVariables
import pandas as pd
import simpy
import order as o
import station as s
import multiprocessing
import time
import warnings
warnings.filterwarnings("ignore")


class Simulation:
    """
    In this class the Simulation calls the order logic from order.py
    """

    def __init__(self, env, order, SIM_TIME):
        self.env = env
        self.order = order
        self.SIM_TIME = SIM_TIME

    def run(self, sim_round):
        GlobaleVariables.SIM_ROUND = sim_round

        self.env.process(self.order.generate_orders())

        self.env.run(until=self.SIM_TIME)


# Create environment
env = simpy.Environment()

station1 = s.Station(env, 1)
station2 = s.Station(env, 2)
station3 = s.Station(env, 3)

GlobaleVariables.stations_list.append(station1)
GlobaleVariables.stations_list.append(station2)
GlobaleVariables.stations_list.append(station3)

# Runtime
SIM_TIME = GlobaleVariables.SIM_TIME

# Only for linear and exponential
df = pd.read_excel(r'best_models_all_utis.xlsx')
df = df.loc[df['utilization'] == GlobaleVariables.SCENARIO]
GlobaleVariables.best_models = df.loc[df['model'] == GlobaleVariables.model]

# Create instance of class Order
order = o.Order(env, 1, 1, 1, 1)


def run_simulation(sim_round):
    """
    This function calls the simulation and runs it.
    :param sim_round: The number of the simulation replication.
    """
    simulation = Simulation(env, order, SIM_TIME)
    simulation.run(sim_round)


start = time.time()
if __name__ == '__main__':
    multiprocessing.freeze_support()

    processes = []
    for i in range(1, 6):
        p = multiprocessing.Process(target=run_simulation, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end = time.time()

    print(f"Dauer: {(end - start) / 60} Minuten")
