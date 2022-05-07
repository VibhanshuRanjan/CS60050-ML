# Roll 18EE35032
# Name Vibhanshu Ranjan
# Assignment Number 3

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def K_Means(data_set, k, iterations):
	clusters = [x+1 for x in range(k)]
	idx = random.sample(range(len(data_set)), k) # selecting k random indices from the data_set itself which will act as starting centroid
	centroid = {} # dictonary with key as cluster number and value as the centroid point
	for i in range(len(idx)):
		centroid[clusters[i]] = data_set.iloc[idx[i]]
	# print(centroid)

	data_labels = [0]*len(data_set) # initializing label/cluster for each data point equal to 0

	for i in range(iterations):
		centroid_t = {}  # keep sum of all points for each cluster/label
		cnt={} # no of points for each cluster/label
		for j in range(len(data_set)):
			euc_dist=[]  # keep euclidean distance of a data point from each cluster/label
			for k in centroid:
				euc_dist.append((np.linalg.norm(data_set.iloc[j]-centroid[k]),k))
			euc_dist.sort(key = lambda x: x[0]) # sorting according to euclidean distance and then updating label of that point
			data_labels[j]=euc_dist[0][1]
			if data_labels[j] not in centroid_t:
				centroid_t[data_labels[j]] = data_set.iloc[j]
				cnt[data_labels[j]] = 1
			else:
				# print(tup_t[data_labels[j]])
				# print(data_set.iloc[j])
				centroid_t[data_labels[j]] +=data_set.iloc[j]
				# print(tup_t[data_labels[j]])
				cnt[data_labels[j]] +=1
				# print(cnt[data_labels[j]])
		# Updating Centroid
		for i in centroid_t:
			centroid[i] = centroid_t[i]/cnt[i]
	return(data_labels)







if __name__ == "__main__":
	data_set = pd.read_csv('Project3.csv')
	data_set = data_set.drop(['ID'] ,axis=1)


	## Normalizing columns of dataset
	for column in data_set:
		if data_set[column].max() > 1:
			data_set[column] = data_set[column ] /data_set[column].max()



	 ## Finding optimal value of K using elbow method
	cost = []
	for i in range(1, 15):
		KM = KMeans(n_clusters=i, max_iter=500)
		KM.fit(data_set)

		# calculates squared error
		# for the clustered points
		cost.append(KM.inertia_)

		# plot the cost against K values
	plt.plot(range(1, 15), cost, color='g', linewidth='3')
	plt.xlabel("Value of K")
	plt.ylabel("Squared Error (Cost)")



	# From the plot obtained in elbow method optimal value of k is coming 6
	## Implementing kmeans clustering
	data_labels = K_Means(data_set, k=6 ,iterations=10)
	print(data_labels)

	f = open("18EE35032_P3.out.txt", "w")
	for i in range(len(data_labels)):
		f.write(str(data_labels[i]) + " ")
	f.close()



