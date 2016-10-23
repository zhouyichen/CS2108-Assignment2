# import the necessary packages

from __future__ import print_function
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
import librosa

def sort_and_get_top_five(array):
	return array.argsort()[-5:][::-1]

def do_classification_gmm(feature_data, model):
	logls = np.empty(len(model))
	logls.fill(-np.inf)

	for label_id, label in enumerate(model):
		logls[label_id] = np.exp(np.sum(model[label].score(feature_data)))

	classification_result_ids = sort_and_get_top_five(logls)
	# print classification_result_ids, logls
	# print model.keys()[classification_result_ids[0]]
	return classification_result_ids

def getAcousticFeatures(y, sr, statistics=True):
	# 4. Compute Melspectrogram features from the raw signal.
	feature_spect = librosa.feature.melspectrogram(y=y, sr=sr, fmax=50000)

	# 5. Compute MFCC features from the Melspectrogram.
	mfcc = librosa.feature.mfcc(S=librosa.logamplitude(feature_spect), n_mfcc=20)
	mfcc_delta = librosa.feature.delta(mfcc)
	mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

	# 6. Compute Zero-Crossing features from the raw signal.
	feature_zerocrossing = librosa.feature.zero_crossing_rate(y=y)
	z_delta = librosa.feature.delta(feature_zerocrossing)
	z_delta2 = librosa.feature.delta(feature_zerocrossing, order=2)

	# 7. Compute Root-Mean-Square (RMS) Energy for each frame.
	feature_energy = librosa.feature.rmse(y=y)
	e_delta = librosa.feature.delta(feature_energy)
	e_delta2 = librosa.feature.delta(feature_energy, order=2)

	feature_matrix = np.vstack((mfcc, mfcc_delta, mfcc_delta2,
		feature_zerocrossing, z_delta, z_delta2,
		feature_energy, e_delta, e_delta2)).T
	
	feature_matrix = feature_matrix[1:-1, :]
	if statistics:
		return {
			'feat': feature_matrix,
			'stat': {
				'mean': np.mean(feature_matrix, axis=0),
				'std': np.std(feature_matrix, axis=0),
				'N': feature_matrix.shape[0],
				'S1': np.sum(feature_matrix, axis=0),
				'S2': np.sum(feature_matrix ** 2, axis=0),
			}
		}
	else:
		return feature_matrix

def getAcousticFeaturesFromPath(audio_reading_path, statistics=True):
	# 1. Load the audio clip;
	y, sr = librosa.load(audio_reading_path)
	# y = extend_to_length_k(y, AUDIO_LENGTH)

	# 2. Separate harmonics and percussives into two waveforms.
	# y_harmonic, y_percussive = librosa.effects.hpss(y)

	# # 3. Beat track on the percussive signal.
	# tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

	return getAcousticFeatures(y, sr, statistics)

def return_all_probabilities(feature_data, model):
	logls = np.empty(len(model))
	logls.fill(-np.inf)

	for label_id, label in enumerate(model):
		logls[label_id] = np.exp(np.sum(model[label].score(feature_data)))

	# classification_result_ids = sort_and_get_top_five(logls)
	return logls / max(logls)


class AcousticSeacher:
	def __init__(self, model, normalizer):
		# store our index path
		self.model = model
		self.normalizer = normalizer

	def search(self, query_audio_path):
		features = getAcousticFeaturesFromPath(query_audio_path, statistics=False)
		features = self.normalizer.normalize(features)
		result = return_all_probabilities(features, self.model)
		return result
