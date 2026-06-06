# metrics.py

import numpy as np

def regret_curve(history,
                 optimum):

    history = np.array(history)

    return optimum - history