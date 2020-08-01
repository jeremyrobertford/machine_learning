import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')


class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):
        self.data = data
        optimum_dict = {}

        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]

        self.max_feature_value = max([np.max(self.data[x]) for x in self.data])
        self.min_feature_value = min([np.min(self.data[x]) for x in self.data])

        # support vectors yi(xi.w+b) = 1
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001, ]

        # extremely expensive
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:
                data_range = self.max_feature_value * b_multiple
                for b in np.arange(-1 * data_range,
                                   data_range,
                                   step * b_multiple):
                    for t in transforms:
                        w_t = w * t
                        found_option = True
                        # Expensive point of SVM model: cycling through
                        # all the data.
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                # Check if point is support vector
                                # yi(xi.w+b) >= 1
                                if not yi*(np.dot(w_t, xi)+b) >= 1:
                                    found_option = False

                        if found_option:
                            # { ||w||: [w, b] }
                            w_mag = np.linalg.norm(w_t)
                            optimum_dict[w_mag] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step

            norms = sorted(optimum_dict.keys())
            # ||w|| : [w, b]
            optimum_choice = optimum_dict[norms[0]]
            self.w = optimum_choice[0]
            self.b = optimum_choice[1]
            latest_optimum = optimum_choice[0][0] + step * 2

    def predict(self, features):
        # sign(xi.w+b)
        # c = classification
        c = np.sign(np.dot(np.array(features), self.w)+self.b)

        if c != 0 and self.visualization:
            self.ax.scatter(features[0], features[1],
                            s=200, marker='*', c=self.colors[c])
        else:
            print('featureset', features, 'is on the decision boundary')

        return c

    def visualize(self):
        for i in data_dict:
            for x in data_dict[i]:
                self.ax.scatter(x[0], x[1], s=100, color=self.colors[i])

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x, w, b, v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # Create points for hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        # Create line
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # (w.x+b) = -1
        # Create points for hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        # Create line
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # (w.x+b) = 0
        # Create points for hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        # Create line
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


if __name__ == '__main__':
    data_dict = {-1: np.array([[1, 7],
                              [2, 8],
                              [3, 8], ]),

                 1: np.array([[5, 1],
                             [6, -1],
                             [7, 3], ])}

    svm = Support_Vector_Machine()
    svm.fit(data=data_dict)
    svm.visualize()
    new_data = [[0, 10],
                [1, 3],
                [3, 4],
                [3, 5],
                [5, 5],
                [5, 6],
                [6, -5],
                [5, 8]]

for p in predict_us:
    print(p, svm.predict(p))
