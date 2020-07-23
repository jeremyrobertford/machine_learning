import numpy as np
import pandas as pd

class LinearRegression:

    def __init__(self, labels=None):


        if labels:
            if len(labels) == X.shape[1]:
                self.labels = labels
            else:
                raise ValueError(('Labels was not the correct length.'
                                  + 'Length given %d != Length expected %d.') 
                                  % (X.shape[1], len(labels)))

    def fit(self, X, y):
        
        # Reshape 1D array so every index is an array with one value.
        # Prevents having to write specific code for 1D arrays.
        if len(X.shape) == 1:
            X = X.reshape((-1,1))
        if len(y.shape) == 1:
            y = y.reshape((-1,1))
        
        self.X = X
        self.y = y
        
        # Append a column of ones to X
        ones = np.ones((self.X.shape[0],1))
        x = np.hstack((ones,self.X))
        xt = x.transpose()
        xtx = np.dot(xt,x)
        xi = np.linalg.inv(xtx)
        xty = np.dot(xt,self.y)
        
        # (X'X)^-1 * X'Y as 1D array
        self.coeffs = np.dot(xi,xty).reshape(-1)
        return self.coeffs

    def predict(self, X):
        return [np.sum(self.coeffs[1:] * x) + self.coeffs[0] for x in X]
    
    def squared_error(self, predicts):
        return np.sum((self.y - predicts)**2) 

    def score(self):

        predicts = self.predict(self.X)
        SE_predicts = self.squared_error(predicts)

        y_mean = self.y.mean()
        y_means = [ y_mean for y in self.y ]
        SE_mean = self.squared_error(y_means)
       
        self.r_sq = (SE_predicts / SE_mean) - 1
        return self.r_sq

    def __repr__(self):
        
        c = {}
        if hasattr(self,'labels'):
            labels = self.labels[::-1]
            labels.append('Intercept')
            labels = labels[::-1]
        else:
            labels = ['b'+str(i) for i in range(len(self.coeffs))]
        
        for i in range(len(self.coeffs)):
            c[labels[i]] = self.coeffs[i]

        return repr(coeff_dict)

    def __str__(self):
        
        s = str(self.coeffs[0])
        if hasattr(self,'labels'):
            labels = self.labels[::-1]
            labels.append('Intercept')
            labels = labels[::-1]
        else:
            labels = ['b'+str(i) for i in range(len(self.coeffs))]
        
        for i in range(1,len(self.coeffs)):
            s += ' + ' + str(self.coeffs[i]) + ' ' + labels[i]

        return s



if __name__ == '__main__':

    X = np.array([4.0,4.5,5.0,5.5,6.0,6.5,7.0], dtype=np.float64)
    y = np.array([33,42,45,51,53,61,62], dtype=np.float64)

    clf = LinearRegression()
    clf.fit(X, y)
    accuracy = clf.score()

    print('Single')
    print('Mr.coeffs:\n', clf.coeffs)
    print('Mr.r_sq:\n', clf.r_sq)
    print('Mr predictions:\n', clf.predict(clf.X))
    print('Print Mr:\n', clf)

    data = pd.DataFrame({'Student':[1,2,3,4,5,6,7,8,9,10],
                         'IQ':[125,104,110,105,100,100,95,95,85,90],
                         'Study Hours':[30,40,25,20,20,20,15,10,0,5],
                         'Test Score':[100,95,92,90,85,80,78,75,72,65]
                         })
    label = 'Test Score'
    X = data.drop(label,'columns').values
    y = data[label].values

    clf = LinearRegression(labels=['Student', 'IQ', 'Study Hours'])
    clf.fit(X, y)
    accuracy = clf.score()

    print()
    print('Multi')
    print('Mr.coeffs:\n', clf.coeffs)
    print('Mr.r_sq:\n', clf.r_sq)
    print('Mr predictions:\n', clf.predict(clf.X))
    print('Print Mr:\n', clf)
    
