from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
# Creating FastAPI instance

 
# Loading Iris Dataset
iris = load_iris()
print(iris)
 
# Getting our Features and Targets
X = iris.data
Y = iris.target
 
# Creating and Fitting our Model
clf = GaussianNB()
clf.fit(X,Y)

test_data = [[4.9, 3. , 1.4, 0.2]]
     
    # Predicting the Class
class_idx = clf.predict(test_data)[0]
     
    # Return the Result
print(iris.target_names[class_idx])
print("golu")