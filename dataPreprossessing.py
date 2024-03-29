import csv
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

filname = 'dataset/train.csv'
x = []
y = []

with open(filname) as csvFile:
    csvReader = csv.reader(csvFile)
    n = 0
    for row in csvReader:
        if n == 0:
            pass

        else:
            y.append(int(row[0]))
            x_in = list(map(int, row[1:]))
            x.append(x_in)
        n += 1


X = np.asarray(x)
X = np.reshape(X, (len(x), 28, -1, 1))

Y = np.asarray(y)
Y = np.reshape(Y, (-1, 1))
onehotencoder = OneHotEncoder()
Y = onehotencoder.fit_transform(Y).toarray()
print('len(x): ', len(x), 'len(x[0]): ', len(
    x[0]), "y[56]: ", y[56], "Y[56]: ", Y[56])
print(X.shape)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()
