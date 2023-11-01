from federated_private_decision_tree import FederatedPrivateDecisionTree
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def simulation_plots(datasets, dataset_names, max_depths, privacy_budgeds, fix_depth=10, fix_epsilon=1, k=10):
    counter = 0
    for data in datasets:
        X = data[0]
        y = data[1]
        # create train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        exp_accuracies = []
        laplace_accuracies = []
        no_dp_accuracies = []
        for depth in max_depths:
            # Calculate the exp_model k times and take the average accuracy
            accuracy = 0
            for i in range(k):
                exp_model = FederatedPrivateDecisionTree(max_depth=depth, privacy_budget=fix_epsilon, diff_privacy=True, leaf_mechanism='exponential')
                exp_model.fit(X_train, y_train)
                predictions = exp_model.predict(X_test)
                exp_accuracy = accuracy_score(y_test, predictions)
                accuracy += exp_accuracy
            accuracy /= k
            exp_accuracies.append(accuracy)

            # Calculate the laplace_model k times and take the average accuracy
            accuracy = 0
            for i in range(k):
                laplace_model = FederatedPrivateDecisionTree(max_depth=depth, privacy_budget=fix_epsilon, diff_privacy=True, leaf_mechanism='laplace')
                laplace_model.fit(X_train, y_train)
                predictions = laplace_model.predict(X_test)
                laplace_accuracy = accuracy_score(y_test, predictions)
                accuracy += laplace_accuracy
            accuracy /= k
            laplace_accuracies.append(accuracy)

            no_dp_model = FederatedPrivateDecisionTree(max_depth=depth, privacy_budget=fix_epsilon, diff_privacy=False)
            no_dp_model.fit(X_train, y_train)
            predictions = no_dp_model.predict(X_test)
            no_dp_accuracy = accuracy_score(y_test, predictions)
            no_dp_accuracies.append(no_dp_accuracy)

        plt.figure()
        plt.plot(max_depths, exp_accuracies, '-', label='exponential')
        plt.plot(max_depths, laplace_accuracies, '-', label='laplace')
        plt.plot(max_depths, no_dp_accuracies, '-', label='no differential privacy')
        plt.title("Accuracy vs depth on " + dataset_names[counter])
        plt.suptitle("epsilon = " + str(fix_epsilon))
        plt.xlabel("depth")
        plt.ylabel("accuracy")
        plt.legend()
        #plt.show()
        # safe plot in plots folder
        name = "plots/accuracy_vs_depth_" + dataset_names[counter].replace(" ", "_") + ".png"
        plt.savefig(name)


        # repeat the same but now take privacy budged on x-axis
        exp_accuracies = []
        laplace_accuracies = []
        no_dp_accuracies = []
        
        for privacy_budget in privacy_budgeds:
            # Calculate the exp_model k times and take the average accuracy
            accuracy = 0
            for i in range(k):
                exp_model = FederatedPrivateDecisionTree(max_depth=fix_depth, privacy_budget=privacy_budget, diff_privacy=True, leaf_mechanism='exponential')
                exp_model.fit(X_train, y_train)
                predictions = exp_model.predict(X_test)
                exp_accuracy = accuracy_score(y_test, predictions)
                accuracy += exp_accuracy
            accuracy /= k
            exp_accuracies.append(accuracy)

            # Calculate the laplace_model k times and take the average accuracy
            accuracy = 0
            for i in range(k):
                laplace_model = FederatedPrivateDecisionTree(max_depth=fix_depth, privacy_budget=privacy_budget, diff_privacy=True, leaf_mechanism='laplace')
                laplace_model.fit(X_train, y_train)
                predictions = laplace_model.predict(X_test)
                laplace_accuracy = accuracy_score(y_test, predictions)
                accuracy += laplace_accuracy
            accuracy /= k
            laplace_accuracies.append(accuracy)

            no_dp_model = FederatedPrivateDecisionTree(max_depth=fix_depth, privacy_budget=privacy_budget, diff_privacy=False)
            no_dp_model.fit(X_train, y_train)
            predictions = no_dp_model.predict(X_test)
            no_dp_accuracy = accuracy_score(y_test, predictions)
            no_dp_accuracies.append(no_dp_accuracy)

        plt.figure()
        plt.plot(privacy_budgets, exp_accuracies, '-', label='exponential')
        plt.plot(privacy_budgets, laplace_accuracies, '-', label='laplace')
        plt.plot(privacy_budgets, no_dp_accuracies, '-', label='no differential privacy')
        plt.title("Accuracy vs privacy budget on " + dataset_names[counter])
        # create text with dataset name outside the plot
        plt.suptitle("depth = " + str(fix_depth))
        plt.xlabel("privacy budget")
        plt.ylabel("accuracy")
        plt.legend()
        #plt.show()
        # safe plot in plots folder
        # create name indicating accuracy vs privacy budged as well as dataset name 
        name = "plots/accuracy_vs_privacy_budget_" + dataset_names[counter].replace(" ", "_") + ".png"
        plt.savefig(name)
        counter += 1
    return


# import breast cancer dataset from sklearn
from sklearn.datasets import load_breast_cancer
breast_data = load_breast_cancer()
X_breast = breast_data.data
y_breast = breast_data.target


# load disease symptom dataset
data = pd.read_csv('data/Disease_symptom_and_patient_profile_dataset.csv')
# drop NA values
data = data.dropna()
# change outcome variable into 0 and one. It is called Outcome Variable and the classes are called Positive and Negative
data['Outcome Variable'] = data['Outcome Variable'].map({'Positive': 1, 'Negative': 0})
# one hot encode the dataset
data = pd.get_dummies(data, drop_first=False)
# change bool to int 0 or one for all bool columns
data = data.astype(float)
X = data.drop('Outcome Variable', axis=1)
y = data['Outcome Variable']
# change X and y to numpy format
X_disease = X.to_numpy()
y_disease = y.to_numpy()
# store y as ints
y_disease = y_disease.astype(int)

# set up parameters for simulation
k = 10 # number of times to repeat the experiment to have a less noisy estimate of accuracy when using differential privacy
fix_epsilon = 1 # privacy budged used when varying depths
fix_depth = 10 # depth used when varying privacy budget
max_depths = [i for i in range(1, 20)]
privacy_budgets = np.linspace(0.1, 2, 10)
datasets = [[X_breast, y_breast], [X_disease, y_disease]]
dataset_names = ["breast cancer dataset", "disease symptom dataset"]

simulation_plots(datasets=datasets, dataset_names=dataset_names, max_depths = max_depths, privacy_budgeds=privacy_budgets, fix_depth=fix_depth, fix_epsilon=fix_epsilon, k=k)


# compare my decision tree without differential privacy vs sklearn decision tree with same parameters on the disease set
X_train, X_test, y_train, y_test = train_test_split(X_disease, y_disease)
my_model = FederatedPrivateDecisionTree(max_depth=fix_depth, diff_privacy=False)
my_model.fit(X_train, y_train)
predictions = my_model.predict(X_test)
my_accuracy = accuracy_score(y_test, predictions)
print("My accuracy: " + str(my_accuracy))
# compare with sklearn decision tree
from sklearn.tree import DecisionTreeClassifier
sklearn_model = DecisionTreeClassifier(max_depth=fix_depth)
sklearn_model.fit(X_train, y_train)
predictions = sklearn_model.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, predictions)
print("Sklearn accuracy: " + str(sklearn_accuracy))
print("Done!")






