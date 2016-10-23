from __future__ import print_function
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.externals import joblib
from common import *

def sort_and_get_top_five(array):
	return array.argsort()[-5:][::-1]

def apply_lable_to_images():
	print('reading data')
	X_train_raw = np.genfromtxt(dp_train_raw_data, dtype=str, delimiter=',')
	# X_test = np.loadtxt(dp_test_raw_data)
	X_train_data = X_train_raw[:, 1:].astype('f')

	# pca = PCA(n_components=400, svd_solver='randomized', whiten=True).fit(X_train_data)
	# save_data('models/visual_pca.pkl', pca)

	# X_train_data = pca.transform(X_train_data)

	number_of_samples = X_train_data.shape[0]

	dtypes = [('id', 'uint64'), ('lable', "int")]
	training_lables = np.loadtxt('data/vine-venue-training.txt', dtype=dtypes)

	lables_table = {}

	print('forming table')
	for i in training_lables:
		lables_table[i[0]] = i[1]

	print('find lable')
	y_train = np.zeros(number_of_samples)
	for i in range(number_of_samples):
		image_id = int(X_train_raw[i, 0][:19])
		y_train[i] = lables_table[image_id]

	print('saving')
	save_data(visual_trian_data, [y_train, X_train_data])


	print('reading data')
	X_test_raw = np.genfromtxt(dp_test_raw_data, dtype=str, delimiter=',')
	# X_test = np.loadtxt(dp_test_raw_data)
	X_test_data = X_test_raw[:, 1:].astype('f')
	# X_test_data = pca.transform(X_test_data)

	number_of_samples = X_test_data.shape[0]

	dtypes = [('id', 'uint64'), ('lable', "int")]
	testing_lables = np.loadtxt('data/vine-venue-validation.txt', dtype=dtypes)

	lables_table = {}
	print('forming table')
	for i in testing_lables:
		lables_table[i[0]] = i[1]

	print('find lable')
	y_test = np.zeros(number_of_samples)
	for i in range(number_of_samples):
		image_id = int(X_test_raw[i, 0][:19])
		y_test[i] = lables_table[image_id]

	print('saving')
	save_data(visual_test_data, [y_test, X_test_data])

def extract_testing_data_for_evaluation():
	print('reading data')
	X_test_raw = np.genfromtxt(dp_test_raw_data, dtype=str, delimiter=',')
	# X_test = np.loadtxt(dp_test_raw_data)
	X_test_data = X_test_raw[:, 1:].astype('f')
	# pca = load_data('models/visual_pca.pkl')
	# X_test_data = pca.transform(X_test_data)

	number_of_samples = X_test_data.shape[0]

	dtypes = [('id', 'uint64'), ('lable', "int")]
	testing_lables = np.loadtxt('data/vine-venue-validation.txt', dtype=dtypes)
	lables_table = {}
	print('forming table')
	for i in testing_lables:
		lables_table[i[0]] = i[1]

	print('find lable')
	y_test = np.zeros(number_of_samples)
	previous_id = 0
	X_combined = []
	features = None
	for i in range(number_of_samples):
		print(i)
		image_id = int(X_test_raw[i, 0][:19])
		y_test[i] = lables_table[image_id]
		if image_id == previous_id:
			features = np.vstack((features, X_test_data[i]))
		else:
			if i!=0:
				X_combined.append(features)
			features = X_test_data[i]
			previous_id = image_id
	X_combined.append(features)
	print(len(X_combined))
	print('saving')
	save_data('visual_test_combined.pkl', [y_test, X_combined])

def train():
	y_train, X_train = load_data(visual_trian_data)
	y_test, X_test = load_data(visual_test_data)

	pca = PCA(n_components=400, svd_solver='randomized', whiten=True).fit(X_train)
	print(np.sum(pca.explained_variance_ratio_))

	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)

	print("Performing GridSearchCV")
	Cs = [1e1, 1e2, 1e3, 1e4]
	gammas = [0.001, 0.005, 0.01, 0.05]
	res = []

	for C in Cs:
		for gamma in gammas:
			print(C, gamma)
			clf = SVC(C=C, gamma=gamma, kernel='rbf', class_weight='balanced')
			clf = clf.fit(X_train_pca, y_train)
			# joblib.dump(clf, "models/%d_%f_visual.pkl" % (C, gamma))
			# y_train_pred = clf.predict_proba(X_train_pca)
			# y_test_pred = clf.predict_proba(X_test_pca)
			# np.apply_along_axis(sort_and_get_top_five, 1, b)
			y_test_pred = clf.predict(X_test_pca)
			accuracy = np.sum(y_test_pred == y_test) / y_test.size
			print(float(accuracy) / y_test.size)
			res.append([C, gamma, accuracy])

	res.sort(key = lambda i: -i[2])
	print(res)
if __name__ == '__main__':
	apply_lable_to_images()
	extract_testing_data_for_evaluation()



