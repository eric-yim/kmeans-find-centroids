from kmeans_general import kmeans_general_func2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# --Note: If there is 1 n_cluster, will write centroid values to csv. Otherwise, output plot is Cost vs n_clusters
# Typically, good n_cluster is marked by "elbow" on plot. If plot is a smooth curve, then n_cluster depends more on use case.

#============================================Params===========================================================
#Parameters to be modified
filename="C:/Users/ericy/Desktop/Pythons/kMeansTF/data/bullLenDispStart.csv"
write_filename="C:/Users/ericy/Desktop/Pythons/kMeansTF/data/bullLenDispStartCV.csv"
n_iterations=100
min_n_cluster = 10
max_n_cluster = 10
step_n_cluster = 1
#=========================================== end params=======================================================
#initialize
df=pd.read_csv(filename,header=None) #Use Pandas to read csv
sample_values = df.values
print('Data Shape',sample_values.shape)
try_vals = np.arange(min_n_cluster,max_n_cluster+1,step_n_cluster)
whole_costs = np.zeros(len(try_vals))
#End initialize

k = 0
for i in try_vals:
    n_clusters = i
    j_costs,centroid_vals,best_j = kmeans_general_func2(sample_values,n_clusters, n_iterations)
    whole_costs[k]= best_j
    k=k+1
    if len(try_vals)==1:
        # print(centroid_vals)
        df = pd.DataFrame(data=np.array(centroid_vals))
        df.to_csv(write_filename,header=False,index=False) 
plt.plot(try_vals,whole_costs,'o')
plt.show()
