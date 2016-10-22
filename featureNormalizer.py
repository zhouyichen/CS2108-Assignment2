from common import *
import cPickle as pickle

class FeatureNormalizer(object):
	"""Feature normalizer class

	Accumulates feature statistics

	Examples
	--------

	>>> normalizer = FeatureNormalizer()
	>>> for feature_matrix in training_items:
	>>>     normalizer.accumulate(feature_matrix)
	>>>
	>>> normalizer.finalize()

	>>> for feature_matrix in test_items:
	>>>     feature_matrix_normalized = normalizer.normalize(feature_matrix)
	>>>     # used the features

	"""
	def __init__(self, feature_matrix=None):
		"""__init__ method.

		Parameters
		----------
		feature_matrix : np.ndarray [shape=(frames, number of feature values)] or None
			Feature matrix to be used in the initialization

		"""
		if feature_matrix is None:
			self.N = 0
			self.mean = 0
			self.S1 = 0
			self.S2 = 0
			self.std = 0
		else:
			self.mean = np.mean(feature_matrix, axis=0)
			self.std = np.std(feature_matrix, axis=0)
			self.N = feature_matrix.shape[0]
			self.S1 = np.sum(feature_matrix, axis=0)
			self.S2 = np.sum(feature_matrix ** 2, axis=0)
			self.finalize()

	def __enter__(self):
		# Initialize Normalization class and return it
		self.N = 0
		self.mean = 0
		self.S1 = 0
		self.S2 = 0
		self.std = 0
		return self

	def __exit__(self, type, value, traceback):
		# Finalize accumulated calculation
		self.finalize()

	def accumulate(self, stat):
		"""Accumalate statistics

		Input is statistics dict, format:

			{
				'mean': np.mean(feature_matrix, axis=0),
				'std': np.std(feature_matrix, axis=0),
				'N': feature_matrix.shape[0],
				'S1': np.sum(feature_matrix, axis=0),
				'S2': np.sum(feature_matrix ** 2, axis=0),
			}

		Parameters
		----------
		stat : dict
			Statistics dict

		Returns
		-------
		nothing

		"""
		self.N += stat['N']
		self.mean += stat['mean']
		self.S1 += stat['S1']
		self.S2 += stat['S2']

	def finalize(self):
		"""Finalize statistics calculation

		Accumulated values are used to get mean and std for the seen feature data.

		Parameters
		----------
		nothing

		Returns
		-------
		nothing

		"""

		# Finalize statistics
		self.mean = self.S1 / self.N
		self.std = np.sqrt((self.N * self.S2 - (self.S1 * self.S1)) / (self.N * (self.N - 1)))

		# In case we have very brain-death material we get std = Nan => 0.0
		self.std = np.nan_to_num(self.std)

		self.mean = np.reshape(self.mean, [1, -1])
		self.std = np.reshape(self.std, [1, -1])

	def normalize(self, feature_matrix):
		"""Normalize feature matrix with internal statistics of the class

		Parameters
		----------
		feature_matrix : np.ndarray [shape=(frames, number of feature values)]
			Feature matrix to be normalized

		Returns
		-------
		feature_matrix : np.ndarray [shape=(frames, number of feature values)]
			Normalized feature matrix

		"""

		return (feature_matrix - self.mean) / self.std

if __name__ == '__main__':
	pkl_file = open(train_mfcc_path, 'rb')
	normalizer = FeatureNormalizer()
	for i in range(train_number):
		print i
		feature_data = pickle.load(pkl_file)['stat']
		normalizer.accumulate(feature_data)
	normalizer.finalize()
	normalizer_file = open(train_mfcc_normalizer_path, 'wb')
	pickle.dump(normalizer, normalizer_file)


