from collections import defaultdict
from math import inf
import random
import csv
import statistics
import numpy as np
import math
from scipy.spatial import distance

def get_centroid(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    (points can have more dimensions than 2)
    
    Returns a new point which is the center of all the points.
    """
    return [statistics.mean(g) for g in points]


def get_centroids(dataset, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.
    Compute the centroid for each of the assigned groups.
    Return `k` centroids in a list
    """
    res = []
    for ass in assignments:
        a = np.array(dataset[i] for i in ass)
        res.append([statistics.mean(g) for g in a])
    return res


def assign_points(data_points, centers):
    """
    """
    assignments = []
    for point in data_points:
        shortest = inf  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    Returns the Euclidean distance between a and b
    """
    return math.dist(a,b)


def distance_squared(a, b):
    return (math.dist(a,b))**2


def cost_function(clustering):
    sum = 0
    for point in clustering:
        sum += distance_squared(point[0], point[1])
    return sum


def generate_k(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    """
    return random.sample(dataset, k)


def generate_k_pp(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    where points are picked with a probability proportional
    to their distance as per kmeans pp
    """
    centers = []
    X = np.array(X)
    
    # Sample the first point
    initial_index = np.random.choice(range(X.shape[0]), )
    centers.append(X[initial_index, :].tolist())
    
    print('max: ', np.max(np.sum((X - np.array(centers))**2)))
    
    # Loop and select the remaining points
    for i in range(k - 1):
        print(i)
        dist = distance(X, np.array(centers))
        
        if i == 0:
            pdf = dist/np.sum(dist)
            centroid_new = X[np.random.choice(range(X.shape[0]), replace = False, p = pdf.flatten())]
        else:
            # Calculate the distance of each point from its nearest centroid
            dist_min = np.min(dist, axis = 1)
        pdf = dist_min/np.sum(dist_min)
# Sample one point from the given distribution
        centroid_new = X[np.random.choice(range(X.shape[0]), replace = False, p = pdf)]
            
        centers.append(centroid_new.tolist())
        
    return np.array(centers)


def _do_lloyds_algo(dataset, k_points):
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = get_centroids(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    clustering = defaultdict(list)
    for assignment, point in zip(assignments, dataset):
        clustering[assignment].append(point)
    return clustering


def k_means(dataset, k):
    if k not in range(1, len(dataset)+1):
        raise ValueError("lengths must be in [1, len(dataset)]")
    
    k_points = generate_k(dataset, k)
    return _do_lloyds_algo(dataset, k_points)


def k_means_pp(dataset, k):
    if k not in range(1, len(dataset)+1):
        raise ValueError("lengths must be in [1, len(dataset)]")

    k_points = generate_k_pp(dataset, k)
    return _do_lloyds_algo(dataset, k_points)
