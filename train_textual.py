from __future__ import print_function
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.externals import joblib
from common import *
from sklearn.neural_network import MLPClassifier
import sys

def sort_and_get_top_five(array):
	return array.argsort()[-5:][::-1]

if __name__ == '__main__':
	n = sys.argv[1]
	X_train = np.loadtxt(textual_train_file_name + n + '.vec')
	X_test = np.loadtxt(textual_test_file_name + n + '.vec')


	# pca = PCA(n_components=400, svd_solver='randomized', whiten=True).fit(X_train)
	# print(np.sum(pca.explained_variance_ratio_))
	
	# X_train_pca = pca.transform(X_train)
	# X_test_pca = pca.transform(X_test)
	
	train_ids = np.load(textual_train_ids)
	test_ids = np.load(textual_test_ids)

	dtypes = [('id', 'uint64'), ('lable', "int")]
	training_lables = np.loadtxt('data/vine-venue-training.txt', dtype=dtypes)
	lables_table = {}
	print('forming table')
	for i in training_lables:
		lables_table[i[0]] = i[1]

	y_train = np.zeros(train_number)
	for i in range(train_number):
		y_train[i] = lables_table[int(train_ids[i])]

	testing_lables = np.loadtxt('data/vine-venue-validation.txt', dtype=dtypes)
	lables_table = {}
	print('forming table')
	for i in testing_lables:
		lables_table[i[0]] = i[1]

	y_test = np.zeros(test_number)
	for i in range(test_number):
		y_test[i] = lables_table[int(test_ids[i])]


	print("Performing GridSearchCV")
	Cs = [1e1, 1e2, 1e3, 1e4]
	gammas = [0.001, 0.005, 0.01, 0.05]
	layers_sizes = [
		(100, 100),
		(100, 80),
		(200, 100),
		(200, 150, 100),
	]
	res = []

	# for C in Cs:
	# 	for gamma in gammas:
	for layer_size in layers_sizes:
		print(layer_size)
		clf = MLPClassifier(hidden_layer_sizes=layer_size, max_iter=2000)
		clf = clf.fit(X_train, y_train)
		# joblib.dump(clf, "models/%d_%f_visual.pkl" % (C, gamma))
		# y_train_pred = clf.predict_proba(X_train)
		# y_test_pred = clf.predict_proba(X_test)
		# np.apply_along_axis(sort_and_get_top_five, 1, b)
		y_train_pred = clf.predict(X_train)
		accuracy = np.sum(y_train_pred == y_train) / y_train.size
		print(float(accuracy) / y_train.size)
		res.append([layer_size, accuracy])

	res.sort(key = lambda i: -i[2])
	print(res)
