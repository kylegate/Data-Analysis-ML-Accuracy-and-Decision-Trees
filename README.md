# Data-Analysis-ML-Accuracy-and-Decision-Trees
 This project explores the impact of training set size and decision tree depth on model accuracy. Using Python and Scikit-learn, we create a decision tree with entropy criterion and visualize accuracy trends with varying training set sizes and tree depths. We output the decision tree to a PDF and generate accuracy graphs for analysis.
**Overview**
Accuracy Analysis with Varying Training Set Size
This part of the project measures the impact of different training set sizes on model accuracy. We use a set of machine learning algorithms to train models and evaluate accuracy as a function of training set size. A graph is created to visualize this relationship, demonstrating how accuracy changes with different training set sizes.
Subset Creation and Decision Tree with Entropy Criterion
Here, we create a subset of data using 4 discrete domain attributes, along with an additional attribute derived from the data. This subset is used to build a decision tree with an entropy criterion. The decision tree is visualized and output as a PDF file for further analysis and interpretation.
Accuracy Analysis with Varying Decision Tree Depth
This section explores the impact of varying decision tree depths on accuracy. A graph is created to visualize the relationship between decision tree depth and accuracy. The depth of the tree is varied over at least six different values to understand how model accuracy changes with increasing or decreasing depth.
**Tools and Technologies**
Python: The primary programming language for implementing machine learning algorithms and creating visualizations.
Scikit-learn: A machine learning library used for creating and analyzing decision trees.
Matplotlib/Seaborn: Libraries used for creating visualizations and graphs.
Graphviz: Used for exporting the decision tree to a PDF file.
**Results and Insights**
Accuracy with Training Set Size
From the graph that measures accuracy based on training set size, we observe that accuracy tends to increase with larger training set sizes. However, diminishing returns are observed beyond a certain point, indicating that more data may not always lead to significant accuracy improvements.
Decision Tree with Entropy Criterion
A decision tree created using the entropy criterion provides insights into the decision-making process. The output PDF file shows the tree structure, which helps visualize the splitting decisions made by the tree.
Accuracy with Varying Decision Tree Depth
The graph that measures accuracy based on decision tree depth reveals that accuracy initially increases with greater depth but may plateau or even decrease at extreme depths. This suggests that very deep trees might overfit the data, reducing generalization capability.
