K-Nearest Neighbour (KNN) Classification
About KNN

K-Nearest Neighbour (KNN) is a supervised machine learning algorithm used mainly for classification. It classifies a new data point by finding its K closest data points (neighbours) and assigning the class that occurs most frequently among them.

The distance between points is calculated using Euclidean distance in this project.

Project Overview

This project demonstrates the effect of different values of K on the classification of a new point using a given dataset. The new point Q = (4,4) is classified using K = 1, K = 3, and K = 5.

Dataset

The dataset contains six training points:

P1 = (4,3) → B
P2 = (3,3) → A
P3 = (5,5) → A
P4 = (2,4) → A
P5 = (8,8) → B
P6 = (7,2) → B
Project Workflow
Create the given dataset.
Separate the input features and class labels.
Define the new point Q = (4,4).
Calculate the Euclidean distance between Q and all training points.
Apply KNN with K = 1, 3, and 5.
Display the nearest neighbours for each K value.
Compare the predicted class for different values of K.
KNN Parameters Used
n_neighbors – Number of neighbours (K)
weights – Weight assigned to neighbours
algorithm – Method used to find neighbours
leaf_size – Controls the tree structure
p – Power parameter for distance calculation
metric – Distance metric
n_jobs – Number of CPU cores used
Output

The program displays the distance of each training point from Q, the nearest neighbours for each K value, and the final classification for K = 1, K = 3, and K = 5.

The experiment demonstrates an important property of KNN: changing the value of K can affect the classification result, which is why selecting an appropriate K is important.
