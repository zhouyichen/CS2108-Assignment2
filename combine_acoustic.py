from common import *

def combine_acoustic_features(mfcc, zero_crossing, rms_energy, number):
	feature_length = NUMBER_OF_FRAME * 15
	combined = np.zeros((number, feature_length))
	combined[:, :NUMBER_OF_FRAME * 13] = mfcc
	combined[:, NUMBER_OF_FRAME * 13:NUMBER_OF_FRAME * 14] = zero_crossing
	combined[:, NUMBER_OF_FRAME * 14:] = rms_energy
	return combined

if __name__ == '__main__':

	train_mfcc = np.loadtxt(train_mfcc_path, delimiter=',')
	train_zero_crossing = np.loadtxt(train_zero_crossing_path, delimiter=',')
	train_rms_energy = np.loadtxt(train_rms_energy_path, delimiter=',')

	train_combined = combine_acoustic_features(train_mfcc, train_zero_crossing, train_rms_energy, train_number)
	np.save(train_acoustic_combined_path, train_combined)

	test_mfcc = np.loadtxt(test_mfcc_path, delimiter=',')
	test_zero_crossing = np.loadtxt(test_zero_crossing_path, delimiter=',')
	test_rms_energy = np.loadtxt(test_rms_energy_path, delimiter=',')

	test_combined = combine_acoustic_features(test_mfcc, test_zero_crossing, test_rms_energy, test_number)
	np.save(test_acoustic_combined_path, test_combined)

