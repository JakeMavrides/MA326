# k-Means demo

import numpy as np
import matplotlib.pyplot as plt

# Set the number of data vectors (n) and the dimension of the data space (m)
n = 12  # Example 1
# n = 1000 # random data
m = 2

# Set the number of clusters (k)
k = 4

# Initialize the data - either as in Example 1 or using random data
XData = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
                  [-1/np.sqrt(2), 1/np.sqrt(2)],
                  [1/np.sqrt(2), -1/np.sqrt(2)],
                  [-1/np.sqrt(2), -1/np.sqrt(2)],
                  [1/2, np.sqrt(3)/2],
                  [-1/2, np.sqrt(3)/2],
                  [1/2, -np.sqrt(3)/2],
                  [-1/2, -np.sqrt(3)/2],
                  [np.sqrt(3)/2, 1/2],
                  [-np.sqrt(3)/2, 1/2],
                  [np.sqrt(3)/2, -1/2],
                  [-np.sqrt(3)/2, -1/2]])

# XData = -1 * np.ones((n, m)) + 2 * np.random.rand(n, m)

# Assign each data vector, randomly to one of the k clusters
IndexSet = np.random.randint(1, k + 1, size=n)

# Plot the data
plt.scatter(XData[:, 0], XData[:, 1], s=64, c=IndexSet, cmap='viridis', edgecolors='k', marker='o')
plt.show()

# Create data structures to store the weight vectors for cluster (c), and the
# weight vectors from the previous iteration (cPrev)
c = np.zeros((k, m))
cPrev = np.zeros((k, m))

# The Alternating Minimization Scheme
doneFlag = False

# Keep alternating updates to weight vectors and cluster assignments until weight
# vectors no longer change their locations
while not doneFlag:
    # Update the weight vectors in each cluster via the centroid formula
    for i in range(1, k + 1):
        # Find the indices for all data vectors currently in cluster i
        ClusterIndices = np.where(IndexSet == i)[0]

        # Find the number of data vectors currently in cluster i
        NumVecsInCluster = len(ClusterIndices)

        # Create a data structure to store weight vector for the current cluster
        c[i - 1, :] = np.zeros(m)

        # Update cluster vector using the centroid formula
        for j in range(NumVecsInCluster):
            c[i - 1, :] += XData[ClusterIndices[j], :] / NumVecsInCluster

    # Plot the updated weight vectors for each cluster
    plt.scatter(XData[:, 0], XData[:, 1], s=64, c=IndexSet, cmap='viridis', edgecolors='k', marker='o')
    plt.scatter(c[:, 0], c[:, 1], s=200, c=np.linspace(1, k, k), cmap='viridis', edgecolors='k', marker='X')
    plt.show()

    # Now reassign all data vectors to the closest weight vector (cluster)
    # Create a data structure to store closest weight vector for each data point
    closestCluster = np.zeros(n)

    # Reassign each data vector to the new, closest cluster
    for d in range(n):
        # Store the coordinates of the current data vector
        xD = XData[d, :]

        # Set the minimum distance tracker to be a very large number
        sqDistMin = 1e16

        # Find the closest weight vector (cluster) to the current data vector
        for i in range(1, k + 1):
            sqDist = np.linalg.norm(c[i - 1, :] - xD, ord=2)

            # If the distance is less than the current min, assign the
            # current data vector to this cluster
            if sqDist < sqDistMin:
                closestCluster[d] = i
                sqDistMin = sqDist

    # Update the assignments of the data vectors to their new clusters
    IndexSet = closestCluster.astype(int)

    # Plot the data and the updated weight vectors
    plt.scatter(XData[:, 0], XData[:, 1], s=64, c=IndexSet, cmap='viridis', edgecolors='k', marker='o')
    plt.scatter(c[:, 0], c[:, 1], s=200, c=np.linspace(1, k, k), cmap='viridis', edgecolors='k', marker='X')
    plt.show()

    # Terminate the alternating scheme if the weight vectors are unaltered
    # relative to the previous iteration
    if np.array_equal(c, cPrev):
        doneFlag = True
    else:
        cPrev = np.copy(c)