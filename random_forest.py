from __future__ import print_function
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from common import *

def sort_and_get_top_five(array):
	return array.argsort()[-5:][::-1]

X_train = np.load(train_acoustic_combined_path)
X_test = np.load(test_acoustic_combined_path)

pca = PCA(n_components=PCA_COMPONENTS, svd_solver='randomized', whiten=True).fit(X_train)
print(np.sum(pca.explained_variance_ratio_))

y_train = np.load(training_lables_path)
y_test = np.load(validation_lables_path)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("Performing GridSearchCV")
n_est = [500, 1000, 10000, 100000]
res = []

for n in n_est:
	print(n)
	clf = RandomForestClassifier(n_estimators=n)
	clf = clf.fit(X_train_pca, y_train)
	joblib.dump(clf, "models/%d_zero_crossing.pkl" % (n))
	y_test_pred = clf.predict(X_test_pca)
	accuracy = np.sum(y_test_pred == y_test)
	print(float(accuracy) / y_test.size)
	res.append([n, accuracy])

res.sort(key = lambda i: -i[1])
print(res)

		# print(classification_report(y_train, y_train_pred, digits=5))
		# print(classification_report(y_test, y_test_pred, digits=5))

# if __name__ == '__main__':
# 	extract_for_folder(train_audio_folder, train_acoustic_folder)
# 	# extract_for_folder(test_audio_folder, test_acoustic_folder)
