import pandas as ps
import matplotlib.pyplot as mb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = ps.read_csv('busfare.csv')
X = data.iloc[:, :-1].values
Y = data.iloc[:, 1].values


training_x, test_x, traning_y, test_y = train_test_split(X, Y, test_size=0.3)


reg = LinearRegression()
reg.fit(training_x, traning_y)

prediction = reg.predict(training_x)

mb.scatter(test_x, test_y, color="blue")
mb.plot(test_x, reg.predict(test_x), color="red")
mb.title ("Oil Price vs Bus Fare")
mb.xlabel = ("Oil Price")
mb.ylabel = ("Bus Fare")
mb.show()
