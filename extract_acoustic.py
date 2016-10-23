# __author__ = "xiangwang1223@gmail.com"

# More details: http://librosa.github.io/librosa/tutorial.html#more-examples.

from __future__ import print_function
import librosa
import glob
from common import *
import cPickle as pickle
from searchers.acoustic_searcher import getAcousticFeaturesFromPath
import csv


def extract_for_folder(input_audio_folder, output_folder):
	acoustic_out_file = open(output_folder + acoustic_combined_file_name, 'wb')
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
	# extract_for_folder(train_audio_folder, train_acoustic_folder)
	extract_for_folder(test_audio_folder, test_acoustic_folder)
