from common import *
from sklearn import mixture
from featureNormalizer import FeatureNormalizer
import sys

def run_gmm_for_acoustic(number_of_components):
	normalizer = pickle.load(open(train_acoustic_normalizer_path, 'rb'))
	y_train = np.load(training_lables_path)

	data = {}
	acoustic_pkl_file = open(train_acoustic_combined_path, 'rb')
	print 'loading data'
	for i in range(train_number):
		if not i% 100: print i
		feature_data = pickle.load(acoustic_pkl_file)['feat']
		feature_data = normalizer.normalize(feature_data)
		lable = y_train[i]
		if lable not in data:
			data[lable] = feature_data
		else:
			data[lable] = np.vstack((data[lable], feature_data))
	print len(data)

	print 'training'
	
	n = int(number_of_components)
	models = {}
	for lable in data:
		print lable
		models[lable] = mixture.GaussianMixture(n_components=n, covariance_type='diag', max_iter=150).fit(data[lable])
	save_data(model_folder+number_of_components+acoustic_gmm_models, models)

def run_gmm_for_visual(number_of_components):
	data = {}
	y_train, X_train = load_data(visual_trian_data)

	size = X_train.shape[0]
	print 'loading data'
	for i in range(size):
		if not i% 100: print i
		feature_data = X_train[i]
		lable = y_train[i]
		if lable not in data:
			data[lable] = feature_data
		else:
			data[lable] = np.vstack((data[lable], feature_data))
	print len(data)

	print 'training'
	
	n = int(number_of_components)
	models = {}
	for lable in data:
		print lable
		models[lable] = mixture.GaussianMixture(n_components=n, covariance_type='diag', max_iter=150).fit(data[lable])
	save_data(model_folder+number_of_components+visual_gmm_models, models)

if __name__ == '__main__':
	number_of_components = sys.argv[1]
	run_gmm_for_acoustic(number_of_components)
	# run_gmm_for_visual(number_of_components)

