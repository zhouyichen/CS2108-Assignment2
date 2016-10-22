# __author__ = "xiangwang1223@gmail.com"

# More details: http://librosa.github.io/librosa/tutorial.html#more-examples.

from __future__ import print_function
import librosa
import glob
from common import *
import cPickle as pickle
import csv

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

def getAcousticFeaturesFromPath(audio_reading_path):
	# 1. Load the audio clip;
	y, sr = librosa.load(audio_reading_path)
	# y = extend_to_length_k(y, AUDIO_LENGTH)

	# 2. Separate harmonics and percussives into two waveforms.
	# y_harmonic, y_percussive = librosa.effects.hpss(y)

	# # 3. Beat track on the percussive signal.
	# tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

	return getAcousticFeatures(y, sr)

def extract_for_folder(input_audio_folder, output_folder):
	acoustic_out_file = open(train_acoustic_combined_path, 'wb')
	# mfcc_writer = csv.writer(mfcc_out_file)

	# melspectrogram_out_file = open(output_folder + melspectrogram_file_name, 'w')
	# melspectrogram_writer = csv.writer(melspectrogram_out_file)

	# zero_crossing_out_file = open(output_folder + zero_crossing_file_name, 'w')
	# zero_crossing_writer = csv.writer(zero_crossing_out_file)

	# rms_energy_out_file = open(output_folder + rms_energy_file_name, 'w')
	# rms_energy_writer = csv.writer(rms_energy_out_file)

	audio_ids = []
	for audio_path in glob.glob(input_audio_folder + "/*.wav"):

		audio_id = audio_path[audio_path.rfind("/") + 1:-4]
		print(audio_id)
		audio_ids.append(audio_id)

		features = getAcousticFeaturesFromPath(audio_path)

		pickle.dump(features, acoustic_out_file)

	# filenames_out_file = open(output_folder + filenames_file_name, 'w')
	# filenames_writer = csv.writer(filenames_out_file)
	# filenames_writer.writerow(audio_ids)

if __name__ == '__main__':
	extract_for_folder(train_audio_folder, train_acoustic_folder)
	extract_for_folder(test_audio_folder, test_acoustic_folder)
