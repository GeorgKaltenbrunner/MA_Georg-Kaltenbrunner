from scipy.stats import expon
import numpy as np
import GlobaleVariables

np.random.seed(seed=1)


def proc_time():
    """
    This creates the processing time for each station.
    :return: The processing time.
    """
    processing_time = expon.rvs(scale=90).round()
    if processing_time > 360:
        processing_time = 360
    GlobaleVariables.processing_time_list.append(processing_time)
    return processing_time
