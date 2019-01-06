

import pandas as pd
import matplotlib.pyplot as mb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('googleplaystore.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, -1] = labelencoder_X.fit_transform(X[:, -1])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()


training_x, test_x, traning_y, test_y = train_test_split(X, Y, test_size=0.3)

reg = LinearRegression()
reg.fit(training_x, traning_y)

future_y = reg.predict(test_x)
mb.scatter(test_x, test_y, color="blue")
mb.plot(test_x, future_y, color="red")
mb.title ("Downloads Price vs Category")
mb.xlabel = ("Downloads")
mb.ylabel = ("Category")
mb.show()
