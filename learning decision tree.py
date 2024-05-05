"""
Name:(Kyle Webb)
Date:(4/14/24)
Assignment:(Assignment #12)
Due Date:(4/14/24)
About this project:(Using
steak-survey.zip compute - Create a graph that measures accuracy based upon training set size,Create a Decision Tree
using the criterion of entropy and the subset of data,Create a graph that measures accuracy based upon the depth of
the decision tree. Project inspired by DescisionTreeDataAccuracyExample by Dr. Works)
Assumptions:(TA changes file path for their machine)
All work below was performed by (Kyle Webb)
"""

from sklearn import tree  # For our Decision Tree
import pandas as pd  # For our DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.tree import plot_tree

########################################################################################################################

filepath = r"C:\Users\kylew\OneDrive\Desktop\steak-risk-survey.xlsx"
df = pd.read_excel(filepath)
attributes = ['Do you ever smoke cigarettes?',
              'Do you ever drink alcohol?',
              'Do you ever gamble?',
              'Have you ever been skydiving?',
              ]
df.dropna(inplace=True)  # drop nan vals
data = pd.get_dummies(df[attributes])  # convert to binary data
x = data  # attributes for training
target = 'Consider the following hypothetical situations: <br>In Lottery A, you have a 50% chance of success, with a payout of $100. <br>In Lottery B, you have a 90% chance of success, with a payout of $20. <br><br>Assuming you have $10 to bet, would you play Lottery A or Lottery B?'
y = df[target]  # trained against column
clf = tree.DecisionTreeClassifier()
# test_size= 0.3 means that our test set will be 30% of the train set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)
clf_train = clf.fit(x_train, y_train)  # Train the Decision Tree

NumRuns = 5
TrainingSetSize = []
ScorePer = []
n = 0
# Create a graph that measures accuracy based upon training set size
# Iterates over different training set sizes -> find model's accuracy on varying training data.
for per in range(10, 50, 5):  # varied by 8 values
    TrainingSetSize.append(per * .01)  # varied training set sizes
    ScorePer.append(0)
    for i in range(NumRuns):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=(per * .01), random_state=100)
        clf_train = clf.fit(x_train, y_train)  # model
        pred = clf_train.predict(x_test)  # model prediction
        ScorePer[n] += accuracy_score(y_test, pred)  # accuracy model
    ScorePer[n] /= NumRuns  # overall accuracy average
    n += 1

# plot graph
d = pd.DataFrame({
    'accuracy': pd.Series(ScorePer),
    'training set size': pd.Series(TrainingSetSize)})
plt.plot('training set size', 'accuracy', data=d, label='accuracy')
plt.ylabel('accuracy')
plt.xlabel('training set size')
plt.savefig('accuracy_vs_training_set_size_graph.pdf')
plt.show()

########################################################################################################################
# Create a subset of the data using 4 discrete domain attributes and the attribute created in step 1
X_subset = data.iloc[:, :4]  # using first four attributes
# split a dataset into a random split and test subsets
X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X_subset, y, test_size=.3)
# Create a Decision Tree using the criterion of entropy and the subset of data created in step 2
clf_sub = DecisionTreeClassifier(criterion='entropy')
clf_sub.fit(X_train_sub, y_train_sub)
# export to PDF
sub = plot_tree(clf_sub,
                feature_names=['Do you ever smoke cigarettes?',
                               'Do you ever drink alcohol?',
                               'Do you ever gamble?',
                               'Have you ever been skydiving?'],
                class_names=['Lottery A', 'Lottery B'],
                filled=True, rounded=True)

plt.savefig('entropy_and_subset_tree.pdf')

plt.title('Entropy and Subsets')
plt.xlabel('Entropy')
plt.ylabel('Subset')

########################################################################################################################

depths = [3, 5, 7, 9, 11, 13]  # Varying depth of the decision tree
accuracies_at_depth = []
for depth in depths:
    clf_depth = DecisionTreeClassifier(max_depth=depth)
    clf_depth.fit(x_train, y_train)
    # Evaluate accuracy
    y_pred_depth = clf_depth.predict(x_test)
    accuracy_depth = accuracy_score(y_test, y_pred_depth)
    accuracies_at_depth.append(accuracy_depth)

plt.figure(figsize=(10, 8))

# plot graph
d = pd.DataFrame({
   'accuracy': pd.Series(accuracies_at_depth),
    'depth': pd.Series(depths)})
plt.plot('depth', 'accuracy', data=d, label='accuracy')
plt.title('Accuracy vs Depth')
plt.ylabel('accuracy')
plt.xlabel('depth')
plt.savefig('accuracy_depth_graph.pdf')
plt.show()
########################################################################################################################
