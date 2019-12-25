# --------------
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn import preprocessing 
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


#Importing the mall dataset with pandas
df = pd.read_csv(path)
print(df.describe())
print(df.head())
# Create an array
X = df.iloc[:,[3,4]]
# Using the elbow method to find the optimal number of clusters

# Initialize K-means algorithm
clusters = list(range(1,10))
c_wcss = []
for c in clusters:
    km = KMeans(n_clusters=c,init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(X)
    #clusters = km.cluster_centers_
    # Cluster centers
    #print("Cluster centers are:", clusters)
    #print('='*20)
    # Within cluster sum of squares
    wcss = km.inertia_
    #print("Within cluster sum of squares is:", wcss)
    c_wcss.append(wcss)
plt.plot(clusters,c_wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
# Applying KMeans to the dataset with the optimal number of cluster


best_model = KMeans(n_clusters=6)
best_model.fit(X)

# Visualising the clusters
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.Genre = le.fit_transform(df.Genre)


# Label encoding and plotting the dendogram
import scipy.cluster.hierarchy as sch

# initialize figure and axes
fig1, ax_1 = plt.subplots(figsize=(20,10))

# dendrogram with "ward" linkage
dend = sch.dendrogram(sch.linkage(df, method='complete'), ax=ax_1)

# plot on a figure
ax_1.set_title("Dendrogram")
ax_1.set_xlabel('Spending_score')
ax_1.set_ylabel('Customer data')
plt.show()



