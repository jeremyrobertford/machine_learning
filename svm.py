import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd

style.use('ggplot')

data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8]]),
              1: np.array([[5, 1],
                           [6, -1],
                           [7, 3]])
            }
class SVM:

    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):
        self.data = data
        # { ||w|| = [w, b] }
        opt_dict = {}

        transforms = [[ 1, 1],
                      [-1, 1],
                      [-1,-1],
                      [ 1,-1]]

        all_data = []

        for yi in self.data:
            for features in self.data[yi]:
                for feature in features:
                    all_data.append(feature)

        self.max = max(all_data)
        self.min = min(all_data)
        del all_data

        step_sizes = [self.max * 0.1,
                      self.max * 0.01,
                      self.max * 0.001
                     ]

        # extremely expensive to fine tune
        b_mult = 5

        opt = self.max * 10

        for step in step_sizes:
            w = np.array([opt, opt])
            optimized = False
            while not optimized:
                r = range(-1 * self.max * b_mult, self.max * b_mult, b_mult)
                for b in r:
                    for t in transforms:
                        w_t = w * t
                        found = True
                        # weakest link in the SVM -> iterating over 
                        # the entire dataset
                        # SMO attempts to fix this
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    break
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step
            norms = sorted(list(opt_dict.keys()))
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]

            opt = opt_choice[0][0] + step * 2

    def predict(self, data):
        c = np.sign(np.dot(np.array(data), self.w) + self.b)
        return c


if __name__='__main__':
    pass
