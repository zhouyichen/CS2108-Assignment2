import numpy as np
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from featureNormalizer import FeatureNormalizer
from common import *
import csv
import sys

def clustering(output_file, k):
	normalizer = pickle.load(open(train_acoustic_normalizer_path, 'rb'))

	all_data = None
	acoustic_pkl_file = open(train_acoustic_combined_path, 'rb')
	print 'loading data'
	for i in range(train_number):
		if not i% 100: print i
		feature_data = pickle.load(acoustic_pkl_file)['feat']
		feature_data = normalizer.normalize(feature_data)
		if not i:
			all_data = feature_data
		else:
			all_data = np.vstack((all_data, feature_data))

	# Perform k-means clustering
	print "Perform k-means clustering: ", k
	kmeans = KMeans(n_clusters=k).fit(all_data)
	joblib.dump((kmeans), output_file, compress=3)

	# # Calculate the histogram of features
	# print "Calculate the histogram of features"
	# im_features = np.zeros((train_number, k), "float32")
	# for i in xrange(train_number):
	#     words, distance = kmeans.predict(X)
	#     for w in words:
	#         im_features[i, w] += 1
	#     im_features[i, :] = cv2.normalize(im_features[i, :]).T

	# joblib.dump((image_ids, im_features, k, voc), output_file, compress=3)

if __name__ == "__main__":
	k = int(sys.argv[1])
	clustering("models/acoustic_cluster_" + str(k) + ".pkl", k)
