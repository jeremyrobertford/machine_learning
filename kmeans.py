from collections import defaultdict
import numpy as np
import pandas as pd


class KMeans:
    '''
    K means seeks to find the centers of different classes. Random
    initial points are chosen for the centroids, then each point is
    labeled as belonging to the centroid that it is closest to. The
    centroid is then moved to the middle of all those points. Repeat
    until the centroids don't move.
    '''
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.cents = {}

        # Set starting cents.
        for i in range(self.k):
            self.cents[i] = data[i]

        # Create classifications.
        for i in range(self.max_iter):
            self.cs = defaultdict(list)

            for features in data:
                distances = []
                for cent in self.cents.values():
                    # Calculate euclidean distance
                    distance = np.linalg.norm(np.array(features)
                                              - np.array(cent))
                    distances.append(distance)

                # Add shortest distance to classifications.
                c = distances.index(min(distances))
                self.cs[c].append(features)

            # Keep previous centroids. Calculate new centroids.
            prev_cents, self.cents = (
                    self.cents,
                    self.calc_cent()
                    )

            # Check for tolerance
            optimized = True
            zip(prev_cents.values(), self.cents.values()):

            for p_cent, n_cent in zip(prev_cents.values(), self.cents.values()):
                per_change = (n_cent - p_cent) / p_cent * 100
                if np.sum(per_change) > self.tol:
                    optimized = False
            if optimized:
                break

    def class_avg(self,c):
        return np.average(self.cs[c], axis=0)

    def calc_cent(self):
        avgs = [self.class_avg(c) for c in self.cs]
        return dict(zip(self.cs,avgs))

    def predict(self, data):
        answer = defaultdict(list)
        for features in data:
            distances = []
            for cent in self.cents.values():
                # Calculate euclidean distance
                distance = np.linalg.norm(np.array(features)-np.array(cent))
                distances.append(distance)
            
            # Add shortest distance to classifications.
            c = distances.index(min(distances))
            answer[c].append(features)

        return dict(answer)

    def __repr__(self):
        return repr(self.cents)



if __name__ == '__main__':

    X = np.array([[1,2],
                  [1.5,1.8],
                  [5,8],
                  [8,8],
                  [1,0.6],
                  [9,11]])

    clf = KMeans()
    clf.fit(X)

    predicts = np.array([[9,10],
                         [1,3],
                         [2,1.5]])

    predictions = clf.predict(predicts)

    print(clf)
    print(predictions)
