import simpy


class Station(object):

    # Initialisiere die Class
    def __init__(self, env, number):
        self.env = env
        self.number = number
        self.machine = simpy.Resource(env, 1)

