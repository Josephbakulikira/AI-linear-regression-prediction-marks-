import pandas
import sklearn
import numpy
from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import pyplot as pyplot
import pickle
from matplotlib import style

#fetching data

data = pandas.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"
X = numpy.array(data.drop([predict], 1))
Y = numpy.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

"""
best = 0
for blabla in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    #storing data in a file

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)

    print(accuracy)
    if accuracy > best:
        best = accuracy
        with open("grademodel.pickle", "wb") as fichier:
            pickle.dump(linear, fichier)"""

pickle_in = open("grademodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("coefficient: \n",  linear.coef_)
print("Intercept: \n",  linear.intercept_)

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
#drawing the graph

p ="absences" #G1, G2, studytime, failures, absences
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel(" Final Grade")
pyplot.show()

