# kmeans-find-centroids
K means clustering is a method to categorize samples: https://en.wikipedia.org/wiki/K-means_clustering
This repository contains files to visualize Cost vs Number of Clusters chosen (to help user select appropriate n_clusters).
When user has chosen desired n_clusters, centroid values can be written to csv file.

MainFile.py is the main file to run from. It loads functions from other py files.
User should alter "parameters" section of MainFile to specify location of csv data file, csv write file, n_iterations, and n_clusters values to test.
CSV input format: Each row is a sample. Each column is a feature.
CSV data should already be normalized. (Alternatively files can be altered for feature weighting)
When there is a single value for n_clusters (min and max are the same value), file writes the Centroid Values to csv.

There is no predict feature in these files. For a sample, the category can be found by finding distance from each centroid, then choosing centroid with lowest distance.

