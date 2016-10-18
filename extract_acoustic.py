# __author__ = "xiangwang1223@gmail.com"

# More details: http://librosa.github.io/librosa/tutorial.html#more-examples.

from __future__ import print_function
import moviepy.editor as mp
import librosa
import glob
from common import *
import csv

def extend_to_length_k(y, k):
	r = k / y.size + 1
	res = np.tile(y, r)
	return res[:k]

def getAcousticFeatures(audio_reading_path):
    # 1. Load the audio clip;
    y, sr = librosa.load(audio_reading_path)
    y = extend_to_length_k(y, AUDIO_LENGTH)

    # 2. Separate harmonics and percussives into two waveforms.
    # y_harmonic, y_percussive = librosa.effects.hpss(y)

    # # 3. Beat track on the percussive signal.
    # tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

    # 4. Compute Melspectrogram features from the raw signal.
    feature_spect = librosa.feature.melspectrogram(y=y, sr=sr, fmax=80000)

    # 5. Compute MFCC features from the Melspectrogram.
    feature_mfcc = librosa.feature.mfcc(S=librosa.logamplitude(feature_spect), n_mfcc=13)

    # 6. Compute Zero-Crossing features from the raw signal.
    feature_zerocrossing = librosa.feature.zero_crossing_rate(y=y)

    # 7. Compute Root-Mean-Square (RMS) Energy for each frame.
    feature_energy = librosa.feature.rmse(y=y)

    return feature_mfcc, feature_spect, feature_zerocrossing, feature_energy

def extract_for_folder(input_audio_folder, output_folder):
	mfcc_out_file = open(output_folder + mfcc_file_name, 'w')
	mfcc_writer = csv.writer(mfcc_out_file)

	melspectrogram_out_file = open(output_folder + melspectrogram_file_name, 'w')
	melspectrogram_writer = csv.writer(melspectrogram_out_file)

	zero_crossing_out_file = open(output_folder + zero_crossing_file_name, 'w')
	zero_crossing_writer = csv.writer(zero_crossing_out_file)

	rms_energy_out_file = open(output_folder + rms_energy_file_name, 'w')
	rms_energy_writer = csv.writer(rms_energy_out_file)

	audio_ids = []
	for audio_path in glob.glob(input_audio_folder + "/*.wav"):

		audio_id = audio_path[audio_path.rfind("/") + 1:-4]
		print(audio_id)
		audio_ids.append(audio_id)

		feature_mfcc, feature_spect, feature_zerocrossing, feature_energy = getAcousticFeatures(audio_path)

		mfcc_writer.writerow(feature_mfcc.flatten())
		melspectrogram_writer.writerow(feature_spect.flatten())
		zero_crossing_writer.writerow(feature_zerocrossing.flatten())
		rms_energy_writer.writerow(feature_energy.flatten())

	filenames_out_file = open(output_folder + filenames_file_name, 'w')
	filenames_writer = csv.writer(filenames_out_file)
	filenames_writer.writerow(audio_ids)

if __name__ == '__main__':
	extract_for_folder(train_audio_folder, train_acoustic_folder)
	extract_for_folder(test_audio_folder, test_acoustic_folder)
