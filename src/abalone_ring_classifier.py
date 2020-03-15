"""
COMP30027 Machine Learning
2017 Semester 1
Project 1 - K-Nearest Neighbour Classification
Student Name  : Ivan Ken Weng Chee
Student ID    : 736901
Student Email : ichee@student.unimelb.edu.au
project1.py   : Abalone-2
"""

# Library imports
import csv
import math
import random

"""
Open the file given by the string filename, and return a data set
comprised of the instances in the file, one per line.
"""
def preprocess_data(filename):
    with open(filename, 'r') as file:
        data_set = list(csv.reader(file))
        file.close()
    return data_set

"""
Splits the data set randomly into two training and testing subsets,
the ratio of which is determined by the train_split argument.
"""
def split_data(data_set, train_split):
    train_set = []
    test_set = []
    for row in range(len(data_set)):
        if random.random() < train_split:
            train_set.append(data_set[row])
        else:
            test_set.append(data_set[row])
    return train_set, test_set

"""
Converts numerical string data into floats, and returns 0 or 1 based
on nominal values in the 'Sex' column
"""
def type_convert(data):
    try:
        float(data)
        return float(data)
    except ValueError:
        if data == "M":
            return 0
        elif data == "F":
            return 1
        return 2

"""
Normalises the data set to aid in similarity/distance calculation
"""
def normalise_data(data_set):
    for row in range(len(data_set)):
        for col in range(len(data_set[row])):
            data_set[row][col] = type_convert(data_set[row][col])
    return data_set

"""
Return a score based on calculating the similarity/distance between
the two given instances, according to the similarity/distance metric
defined by the string method.
"""
def compare_instance(instance1, instance2, method):
    if method == "manhattan":
        return(manhattanDistance(instance1, instance2))
    elif method == "hamming":
        return(hammingDistance(instance1, instance2))
    else:
        return(euclideanDistance(instance1, instance2))

"""
Returns a list of (class, score) 2-tuples, for each of the k best
neighbours for the given instance from the test data set, based
on all of the instances in the training data set.
"""
def get_neighbours(instance, training_data_set, k, method):
    epsilon = 0.5
    distances = []
    # Iterate over each instance in the training data set
    for i in range(len(training_data_set)):
        age = determine_class(training_data_set[i][-1])
        dist = 0
        # Calculates based on different distance measures
        dist = compare_instance(instance[:-1], training_data_set[i][:-1], method)
        # Voting strategy
        if (method == "inverse-distance"):
            distances.append((age, 1 / (dist + epsilon)))
        else:
            distances.append((age, dist))
    # Sorts the distances in increasing order and selects the kth instances
    distances.sort(key=get_key)
    neighbours = []
    for i in range(k):
        neighbours.append(distances[i])
    return neighbours

"""
Returns the distance to be compared in sorting
"""
def get_key(item):
    return item[1]

"""
Return a predicted class label, according to the given neighbours
defined by a list of (class, score) 2-tuples, and the chosen
voting method.
"""
def predict_class(neighbours, method):
    votes = {}
    # Creates a dictionary of classes and their respective vote counts
    for i in range(len(neighbours)):
        response = neighbours[i][0]
        if response in votes:
            votes[response] += 1
        else:
            votes[response] = 1
    # Class with the highest vote is the predicted class
    predicted_class = sorted(votes.items(), key=get_key, reverse=True)[0][0]
    return predicted_class

"""
Calculates the confusion matrix and returns evaluation based on metric
"""
def calculate_confusion(test_set, predictions, metric):
    classes = ["young", "old"]
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for c in range(len(classes)):
        for i in range(len(test_set)):
            # True instance
            if determine_class(test_set[i][-1]) == classes[c]:
                # True instance as true (TP)
                if predictions[i] == determine_class(test_set[i][-1]):
                    tp += 1
                # True instance as false (FN)
                else:
                    fn += 1
            # False instance
            else:
                # False instance as true (FP)
                if predictions[i] == classes[c]:
                    fp += 1
                # False instance as false (TN)
                else:
                    tn += 1
    if metric == "precision":
        return 100 * tp / (tp + fp)
    elif metric == "recall":
        return 100 * tp / (tp + fn)
    elif metric == "error":
        return 100 * (1 - ((tp + tn) / (tp + fp + fn + tn)))
    return 100 * (tp + tn) / (tp + fp + fn + tn)

"""
Determines the class of the abalone based on its number of rings
"""
def determine_class(rings):
    if rings < 11:
        return "young"
    return "old"

"""
Returns the calculated value of the evaluation metric, based on
dividing the given data set into training and test splits,
using the preferred evaluation strategy.
"""
def evaluate(data_set, metric):
    data_set = normalise_data(data_set)
    train_set, test_set = split_data(data_set, 0.67)
    predictions = []
    # Classifies each instance in the test set
    for i in range(len(test_set)):
        neighbours = get_neighbours(test_set[i], train_set, 7, "hamming")
        prediction = predict_class(neighbours, "inverse-distance")
        predictions.append(prediction)
    score = calculate_confusion(test_set, predictions, metric)
    return score

"""
Returns the Euclidean distance between two vectors
"""
def euclideanDistance(a, b):
    dist = 0
    for i in range(len(a)):
        dist += pow(a[i] - b[i], 2)
    return math.sqrt(dist)

"""
Returns the Hamming distance between two vectors
"""
def hammingDistance(a, b):
    dist = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            dist += 1
    return dist

"""
Returns the Manhattan distance between two vectors
"""
def manhattanDistance(a, b):
    if len(a) != len(b):
        return - 1
    ans = 0
    for i in range(len(a)):
        ans += abs(a[i] - b[i])
    return ans

"""
Entry point to the program
"""
if __name__ == '__main__':
    print(evaluate(preprocess_data('data/abalone.data.csv'), 'accuracy'))
