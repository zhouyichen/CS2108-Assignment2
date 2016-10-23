from common import *
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from featureNormalizer import FeatureNormalizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import sys

def in_top_five(pred, real):
	X = real.astype(int) - 1
	# print X.shape
	Y = pred.argsort()[:, -5:]
	# print Y[1]
	# print X[1]
	# print Y.shape
	# print X.shape
	# print Y == X[:, None]
	return np.sum((Y == X[:, None]).any(1).astype(int))

def bow_on_data(cluster_data, k):
	normalizer = pickle.load(open(train_acoustic_normalizer_path, 'rb'))
	y_train = np.load(training_lables_path)
	y_test = np.load(validation_lables_path)

	kmeans = joblib.load(cluster_data)

	train_acoustic_pkl_file = open(train_acoustic_combined_path, 'rb')
	X_train = np.zeros((train_number, k), "float32")
	print 'loading data'
	for i in range(train_number):
		if not i% 100: print i
		feature_data = pickle.load(train_acoustic_pkl_file)['feat']
		feature_data = normalizer.normalize(feature_data)
		words = kmeans.predict(feature_data)
		for w in words:
			X_train[i, w] += 1
		X_train[i, :] = cv2.normalize(X_train[i, :]).T

	test_acoustic_pkl_file = open(test_acoustic_combined_path, 'rb')
	X_test = np.zeros((test_number, k), "float32")
	print 'loading data'
	for i in range(test_number):
		if not i% 100: print i
		feature_data = pickle.load(test_acoustic_pkl_file)['feat']
		feature_data = normalizer.normalize(feature_data)
		words = kmeans.predict(feature_data)
		for w in words:
			X_test[i, w] += 1
		X_test[i, :] = cv2.normalize(X_test[i, :]).T

	print("Performing GridSearchCV")
	Cs = [1e2, 1e3, 1e4, 1e5]
	gammas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
	res = []

	for C in Cs:
		for gamma in gammas:
			number = y_test.size
			print(C, gamma)
			# clf = SVC(C=C, gamma=gamma, kernel='rbf', class_weight='balanced')
			# clf = clf.fit(X_train, y_train)
			# y_test_pred = clf.predict(X_test)
			# accuracy = np.sum(y_test_pred == y_test)
			
			clf = SVC(C=C, gamma=gamma, kernel='rbf', class_weight='balanced', probability=True)
			clf = clf.fit(X_train, y_train)
			y_test_pred = clf.predict_proba(X_test)
			accuracy = in_top_five(y_test_pred, y_test)

			# joblib.dump(clf, "models/%d_%d_%f_bow.pkl" % (k, C, gamma))
			print(float(accuracy) / number)
			res.append([C, gamma, accuracy])

	res.sort(key = lambda i: -i[2])
	print(res)

if __name__ == "__main__":
	k = int(sys.argv[1])
	bow_on_data("models/acoustic_cluster_" + str(k) + ".pkl", k)
