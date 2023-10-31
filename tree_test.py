from federated_private_decision_tree import FederatedPrivateDecisionTree
# import breast cancer dataset from sklearn
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target
# create train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
# create and fit model
model = FederatedPrivateDecisionTree(max_depth=10, privacy_budget=0.5)
model.fit(X_train, y_train)
# make predictions
predictions = model.predict(X_test)
# calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: " + str(accuracy))

# compare with decision tree from sklearn
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=10, criterion='entropy')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: " + str(accuracy))

