from sklearn.datasets import load_iris
iris = load_iris()

x = iris.data 
y = iris.target

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.4, random_state = 1)

from sklearn.naive_bayes import GaussianNB 

model = GaussianNB()

model.fit(xtrain, ytrain)

ypred = model.predict(xtest)

from sklearn import metrics 

print(metrics.accuracy_score(ytest, ypred) * 100)
