import cv2
import numpy as np
import cPickle as pickle

AUDIO_LENGTH = 308700
NUMBER_OF_FRAME = 603
NUMBER_OF_CLASSES = 30

class_lables = [str(i+1) for i in range(NUMBER_OF_CLASSES)]

PCA_COMPONENTS = 400
train_number = 3000
test_number = 900

data_folder = 'data/'
model_folder = 'models/'

training_lables_path = data_folder + 'sorted_training_lable.npy'
validation_lables_path = data_folder + 'sorted_validation_lable.npy'

video_folder = data_folder + 'video/'
train_video_folder = video_folder + 'training/'
test_video_folder = video_folder + 'validation/'

frame_folder = data_folder + 'frame/'
train_frame_folder = frame_folder + 'training/'
test_frame_folder = frame_folder + 'validation/'

audio_folder = data_folder + 'audio/'
train_audio_folder = audio_folder + 'training/'
test_audio_folder = audio_folder + 'validation/'

acoustic_folder = data_folder + 'acoustic/'
train_acoustic_folder = acoustic_folder + 'training/'
test_acoustic_folder = acoustic_folder + 'validation/'
train_acoustic_normalizer_path = train_acoustic_folder + 'acoustic_normalizer.pkl'
acoustic_gmm_models = '_acoustic_gmm.pkl'

visual_folder = data_folder + 'visual/'
train_visual_folder = visual_folder + 'training/'
test_visual_folder = visual_folder + 'validation/'

mfcc_file_name = 'mfcc.pkl'
train_mfcc_path = train_acoustic_folder + mfcc_file_name
test_mfcc_path = test_acoustic_folder + mfcc_file_name


melspectrogram_file_name = 'melspectrogram.csv'
train_melspectrogram_path = train_acoustic_folder + melspectrogram_file_name
test_melspectrogram_path = test_acoustic_folder + melspectrogram_file_name

zero_crossing_file_name = 'zero_crossing.csv'
train_zero_crossing_path = train_acoustic_folder + zero_crossing_file_name
test_zero_crossing_path = test_acoustic_folder + zero_crossing_file_name

rms_energy_file_name = 'rms_energy.csv'
train_rms_energy_path = train_acoustic_folder + rms_energy_file_name
test_rms_energy_path = test_acoustic_folder + rms_energy_file_name

acoustic_combined_file_name = 'acoustic_combined.pkl'
train_acoustic_combined_path = train_acoustic_folder + acoustic_combined_file_name
test_acoustic_combined_path = test_acoustic_folder + acoustic_combined_file_name

filenames_file_name = 'filenames.csv'
train_filenames_path = train_acoustic_folder + filenames_file_name
test_filenames_path = test_acoustic_folder + filenames_file_name

textual_folder = data_folder + 'textual/'
textual_train_file_name = textual_folder + 'vine_desc_training_'
textual_test_file_name = textual_folder + 'vine_desc_validation_'

dp_train_raw_data = train_visual_folder + 'dp_raw.csv'
vc_train_raw_data = train_visual_folder + 'vc_raw.csv'

dp_test_raw_data = test_visual_folder + 'dp_raw.csv'
vc_test_raw_data = test_visual_folder + 'vc_raw.csv'

visual_trian_data = data_folder + 'visual_train.pkl'
visual_test_data = data_folder + 'visual_test.pkl'

visual_gmm_models = '_visual_gmm.pkl'

def save_data(filename, data):
    pickle.dump(data, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def load_data(filename):
    return pickle.load(open(filename, "rb"))
