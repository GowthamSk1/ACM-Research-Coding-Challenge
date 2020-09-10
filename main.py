import numpy as np
import pandas as pd  
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

# Parse data from CSV file
data_frame = pd.read_csv('ClusterPlot.csv')
csv_data = pd.DataFrame(data_frame, columns=['V1','V2'])

# Store row and col in numpyarray
row = data_frame['V1'].to_numpy()
col = data_frame['V2'].to_numpy()

# For calculating distortion of K clusters
dis = []

# Store data in np array
X = np.array(list(zip(row, col))).reshape(len(row), 2)
for i in range(1, 7):  # for K clusters calculate the distortion
    model = KMeans(n_clusters=i).fit(X)  
    model.fit(X) 
    dis.append(sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis = 1)) / X.shape[0])
    
''' PLOT OPTIMAL K '''
plt.plot(range(1, 7), dis, 'bx-')
plt.xlabel('k (Number of Clusters)')
plt.ylabel('Variance ')
plt.title('Elbow Method (CLUSTERING) to find the optimal k')
plt.show()

''' PLOT CENTROIDS USING OPTIMAL CLUSTERS '''
kmeans = KMeans(n_clusters=3).fit(csv_data)
cent = kmeans.cluster_centers_
plt.scatter(csv_data['V1'], csv_data['V2'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(cent[:, 0], cent[:, 1], c='red', s=50)
plt.show()
