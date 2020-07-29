from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from pprint import pprint
import random


class OrderError(Exception):
    pass


class KNN:
    '''
    K nearest neighbors is given a point and then calculates the
    euclidean distance between that point and all the all points.
    The k nearest points all contribute a vote. Their vote is equal to
    their class value. The class

            with the highest votes is what the given point's class is.
    '''
    def __init__(self, k=3):
        self.k = k
        return None

    def fit(self, data, unknowns):

        if len(data) >= self.k:
            raise ValueError('K is less than total voting groups.'
                             '{} >= {}'.format((len(data)), self.k))

        self.data = data
        self.unknowns = np.array(unknowns)
        self.answer = self.calc_distances()

        return self.answer

    def calc_distances(self):
        distances = []

        for group in self.data:
            for features in self.data[group]:
                # Calculate euclidean distance
                distance = np.linalg.norm(np.array(features)
                                          - np.array(self.unknowns))
                distances.append([distance, group])

        # Get most common label from k nearest neighbors
        k_neighbors = [i[1] for i in sorted(distances)[:self.k]]
        answer = Counter(k_neighbors).most_common(1)[0][0]

        return answer

    def predict(self, unknowns):
        if hasattr(self, 'data'):
            return self.fit(self.data, unknowns)
        else:
            raise OrderError('Must use fit() before predict().')

    def score(self, train, test):

        correct, total = 0, 0

        for group in test:
            for features in test[group]:

                answer = KNN(k=self.k).fit(train, features)
                if group == answer:
                    correct += 1
                total += 1

        return correct / total

    def __repr__(self):
        return repr(self.answer)


if __name__ == '__main__':

    # Import data
    df = pd.read_csv('..\\data_sets\\breast-cancer-wisconsin.data',
                     header=None)
    df.replace('?', -99999, inplace=True)
    df.drop([0], axis=1, inplace=True)
    df = df.astype(float).values.tolist()

    # Create a train test split
    random.shuffle(df)
    test_size = 0.2
    train_set = defaultdict(list)
    test_set = defaultdict(list)
    train_data = df[:-int(test_size*len(df))]
    test_data = df[-int(test_size*len(df)):]

    # Format datasets for model and remove label column from data.
    train = []
    for i in train_data:
        train_set[i[-1]].append(i[:-1])
        train.append(i[:-1])
    test = []
    for i in test_data:
        test_set[i[-1]].append(i[:-1])
        test.append(i[:-1])

    clf = KNN(k=5)

    # Determine which arrays go to which label.
    test_answers = defaultdict(list)
    for t in test:
        answer = clf.fit(train_set, t)
        test_answers[answer].append(t)
    pprint(test_answers)

    # Doesn't take into account an imbalanced classification set.
    accuracy = clf.score(train_set, test_set)
    print('Accuracy:', accuracy)
