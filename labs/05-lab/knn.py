from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def get_error(predicted, actual):
    return sum(map(lambda x: 1 if (x[0] != x[1]) else 0, zip(predicted, actual))) / len(predicted)


class KNN:

    def __init__(self, k, X_train, y_train):
        self.k = k
        self.X_train = X_train
        self.y_train = y_train
        self.distance_matrix = None

    def train(self):
        self.distance_matrix = NearestNeighbors(n_neighbors=self.k, algorithm='auto').fit(X_train)
        print("distance: " + repr(self.distance_matrix))

    def predict(self, example):
        return self.distance_matrix.predict(example)

    def test(self, test_input, labels):
        actual = labels
        predicted = (self.predict(test_input))
        print("error = ", get_error(predicted, actual))


# Add the dataset here
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Split the data 70:30 and predict.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7)
# create a new object of class KNN
knn_object = KNN(3, X_train, y_train)
# plot a boxplot that is grouped by Species. 
# You may have to ignore the ID column
plt.plot(knn_object)
# predict the labels using KNN
knn_object.train()
print(type(knn_object))
labels = knn_object.predict(X_test)
# use the test function to compute the error
knn_object.test(X_test, labels)
