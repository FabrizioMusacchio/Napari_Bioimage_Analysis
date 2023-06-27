import random
import numpy as np
from scipy.spatial import distance

class KMeansMahalanobis:
    def __init__(self, k, initial_iterations=5, iterations=100):
        self.k = k
        self.initial_iterations = initial_iterations
        self.iterations = iterations
        self.seed = 100
    
    def mahalanobis_distance(self, u, v, cov_inv):
        """Computes the Mahalanobis distance between two arrays.

        Parameters
        ----------
        u: array-like
            First input array.
        v: array-like 
            Second input array.
        cov_inv: ndarray
            The inverse of the covariance matrix.    

        Returns
        -------
        float
            Mahalanobis distance between vectors u and v.
        """
        return np.sqrt(np.dot(np.dot(u - v, cov_inv), (u - v)))
    
    def fit(self, dataset):
        """Computes k means clustering using Mahalanobis distance.

        Parameters
        ----------
        dataset: array-like
            Training data.

        Returns
        -------
            References the instance object. 
        """
        #initialization step
        random.seed(self.seed)
        
        #assign initial centroids
        self.centroids = list(random.sample(list(dataset), 1))
        
        for i in range(self.k - 1):
            dist = np.array([min([np.linalg.norm(sample - centroid)**2 for centroid in self.centroids]) for sample in dataset])
            index = np.random.choice(range(len(dataset)), p=(dist/dist.sum()))
            self.centroids.append(dataset[index])
        
        #assign initial clusters 
        for i in range(self.initial_iterations):
            
            self.partitions = {i:[] for i in range(self.k)}
        
            for sample in dataset:
                distances = [np.sqrt(distance.euclidean(sample, centroid)) for centroid in self.centroids]
                self.partitions[distances.index(min(distances))].append(sample)
            
            #update the centroids
            for cluster in self.partitions:
                self.centroids[cluster] = np.mean(self.partitions[cluster], axis=0)
        
        #iteration step
        for i in range(self.iterations):            
            
            old_partitions = dict(self.partitions)
            
            self.partitions = {i:[] for i in range(self.k)}
            
            #assign points to a cluster given the smallest mahalanobis distance            
            for sample in dataset:
                distances = [self.mahalanobis_distance(np.mean(cluster, axis=0), sample, np.linalg.inv(np.cov(cluster, rowvar=False))) for cluster in old_partitions.values()]
                self.partitions[distances.index(min(distances))].append(sample)
            
            #compare current cluster with the previous one  
            clusters_comparison = [np.allclose(np.mean(self.partitions[cluster]), np.mean(old_partitions[cluster])) for cluster in self.partitions]
            
            if False not in clusters_comparison:
                return self
                break
           
        return self
    
    def predict(self, dataset):
        """Assigns each data point in the dataset to the closest cluster.

        Parameters
        ----------
        dataset: array-like
            New data to be assigned to clusters.

        Returns
        -------
        list
            Cluster index each sample belongs to. 
        """
        self.labels = []
        for sample in dataset:
            distances = [self.mahalanobis_distance(np.mean(cluster, axis=0), sample, np.linalg.inv(np.cov(cluster, rowvar=False))) for cluster in self.partitions.values()]
            self.labels.append(distances.index(min(distances)))
        return self.labels