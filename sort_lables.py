from common import *
import csv

dtypes = [('id', 'uint64'), ('lable', "int")]

training_lables = np.loadtxt('data/vine-venue-training.txt', dtype=dtypes)
training_lables.sort()
lables = [p[1] for p in training_lables]
np.save(training_lables_path, lables)

validation_lables = np.loadtxt('data/vine-venue-validation.txt', dtype=dtypes)
validation_lables.sort()
lables = [p[1] for p in validation_lables]
np.save(validation_lables_path, lables)
