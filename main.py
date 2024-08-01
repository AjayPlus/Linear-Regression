import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# gets data from "student-mat.csv"
# identifies separator as ";"
data = pd.read_csv("student-mat.csv", sep=';')
# narrowed down data
data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]]

predict = "G3"

x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

# taking attributes/labels, and splitting them into 4 different arrays to train the model with
# different data than what is used in the actual model
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# training model
linear = linear_model.LinearRegression()
# find best fit line using x_train data and y_train data
linear.fit(x_train, y_train)
# value with accuracy of model
acc = linear.score(x_test, y_test) * 100
print(acc, "%")

# print('Co: \n', linear.coef_)
# print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
