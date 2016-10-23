# import the necessary packages

from __future__ import print_function
from __future__ import division
import numpy as np
import cv2

import os.path
import re
import csv
import tensorflow as tf
import glob

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
	'model_dir', '/tmp/imagenet',
	"""Path to classify_image_graph_def.pb, """
	"""imagenet_synset_to_human_label_map.txt, and """
	"""imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',   ### here you can indicate the image file !!
						   """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
							"""Display this many predictions.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


def get_key_frames_from_video(vidcap):
	count = 0
	lastHist = None
	sumDiff = []
	frames = []

	while True:
		success, frame = vidcap.read()
		if not success:
			break
		frames.append(frame)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

		if count >0:
			diff = np.abs(hist - lastHist)
			s = np.sum(diff)
			sumDiff.append(s)
		lastHist = hist
		count += 1

	if len(frames) == 0:
		return

	m = np.mean(sumDiff)
	std = np.std(sumDiff)

	candidates = []
	candidates_value = []
	for i in range(len(sumDiff)):
		if sumDiff[i] > m + std*3:
			candidates.append(i + 1)
			candidates_value.append(sumDiff[i])

	if len(candidates) > 20:
		top10list = sorted(range(len(candidates_value)), key=lambda i: candidates_value[i])[-9:]
		res = []
		for i in top10list:
			res.append(candidates[i])
		candidates = sorted(res)

	candidates = [0] + candidates

	keyframes = []
	lastframe = -2
	for frame in candidates:
		if not frame == lastframe + 1:
			keyframes.append(frame)
		lastframe = frame

	results = []
	count = 0
	for frame in keyframes:
		image = frames[frame]
		results.append([count, image])
		count += 1
	return results
	
def getKeyFrames(vidcap, store_frame_path):
	keyframes = get_key_frames_from_video(vidcap)
	for count, image in keyframes:
		cv2.imwrite(store_frame_path+"frame%d.jpg" % count, image)
		
def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
	  FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	_ = tf.import_graph_def(graph_def, name='')

class DeepLearningSearcher(object):
	def __init__(self):

		"""Creates a graph from saved GraphDef file and returns a saver."""
		# Creates graph from saved graph_def.pb.
		with tf.gfile.FastGFile(os.path.join(
			FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def, name='')

		with tf.Session() as sess:
		# Some useful tensors:
		# 'softmax:0': A tensor containing the normalized prediction across
		#   1000 labels.
		# 'pool_3:0': A tensor containing the next-to-last layer containing 2048
		#   float description of the image.
		# 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
		#   encoding of the image.
		# Runs the softmax tensor by feeding the image_data as input to the graph.
			self.softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
			self.sess = sess


	def run_inference_on_images(input_path):
		# Creates graph from saved GraphDef.
		create_graph()
		feature_list = None

		with tf.Session() as sess:
			# Some useful tensors:
			# 'softmax:0': A tensor containing the normalized prediction across
			#   1000 labels.
			# 'pool_3:0': A tensor containing the next-to-last layer containing 2048
			#   float description of the image.
			# 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
			#   encoding of the image.
			# Runs the softmax tensor by feeding the image_data as input to the graph.
			softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

			# use glob to grab the image paths and loop over them
			for imagePath in glob.glob(input_path + "/*.jpg"):

				# extract the image ID (i.e. the unique filename) from the image
				# path and load the image itself
				imageID = imagePath[imagePath.rfind("/") + 1:-4]
				print(imageID)

				# describe the image
				image_data = tf.gfile.FastGFile(imagePath, 'rb').read()
				predictions = sess.run(softmax_tensor,
									 {'DecodeJpeg/contents:0': image_data})
				predictions = np.squeeze(predictions)

				if not feature_list:
					feature_list = predictions
				else:
					feature_list = np.vstack((feature_list, predictions))
		return feature_list
			  

def return_all_probabilities(feature_data, model):
	logls = np.empty(len(model))
	logls.fill(-np.inf)

	for label_id, label in enumerate(model):
		logls[label_id] = np.exp(np.sum(model[label].score(feature_data)))

	# classification_result_ids = sort_and_get_top_five(logls)
	return logls / max(logls)


class VisualSeacher(object):
	def __init__(self, model):
		# store our index path
		self.model = model
		self.dp_searcher = DeepLearningSearcher()

	def search(self, vidcap, folder_name):
		getKeyFrames(vidcap, folder_name)
		features = self.dp_searcher.run_inference_on_images(folder_name).reshape(-1, 1008)
		result = return_all_probabilities(features, self.model)
		print(result)
		return result

