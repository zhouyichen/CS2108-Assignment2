from common import *
from sklearn import mixture
from featureNormalizer import FeatureNormalizer
from searchers.acoustic_searcher import do_classification_gmm
import sys

def evaluation_for_acoustic(number_of_components=64):
	# 64
	models = load_data(model_folder+number_of_components+acoustic_gmm_models)
	
	normalizer = pickle.load(open(train_acoustic_normalizer_path, 'rb'))
	y_test = np.load(validation_lables_path)

	acoustic_pkl_file = open(test_acoustic_combined_path, 'rb')
	correct = 0
	print 'loading data'
	for i in range(test_number):
		feature_data = pickle.load(acoustic_pkl_file)['feat']
		feature_data = normalizer.normalize(feature_data)
		lable = int(y_test[i]) - 1
		result = do_classification_gmm(feature_data, models)
		if lable in result:
			correct += 1

	print number_of_components, correct

def evaluation_for_visual(number_of_components=18):
	# 18
	models = load_data(model_folder+number_of_components+visual_gmm_models)
	
	print 'loading data'
	y_test, X_test = load_data('visual_test_combined.pkl')

	correct = 0

	print 'evaluation'
	for i in range(test_number):
		feature_data = X_test[i].reshape(-1, 1008)
		lable = int(y_test[i]) - 1
		result = do_classification_gmm(feature_data, models)
		if lable in result:
			correct += 1

	print number_of_components, correct

if __name__ == '__main__':
	number_of_components = sys.argv[1]
	evaluation_for_acoustic(number_of_components)
	# evaluation_for_visual(number_of_components)

	