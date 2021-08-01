import csv
import csv
import numpy as np

import os.path

import tensorflow as tf




class Dataset():

	def __init__(self, seq_length=50, class_limit=2):
		self.seq_length = seq_length
		self.class_limit = class_limit
		path = os.getcwd()
		self.sequence_path = os.path.join(path,'data', 'sequence')
		# Get the data.
		self.data = self.get_data()
		# Get the classes.
		self.classes = ["Alert","Drowsy"]

	def get_data(self):
		with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
			reader = csv.reader(fin)
			data = list(reader)

		return data

	def get_class_one_hot(self, class_str):
		# Encode it first.
		label_encoded = self.classes.index(class_str)

		# Now one-hot it.
		label_hot = tf.keras.utils.to_categorical(label_encoded, len(self.classes))

		assert len(label_hot) == len(self.classes)

		return label_hot

	def get_all_sequences_in_memory(self, train_test):
		print("Loading samples into memory for --> ",train_test)

		X, y = [], []
		for videos in self.data:
			if(videos[0] == train_test):
				sequence = self.get_extracted_sequence(videos)
				if sequence is None:
					print("Can't find sequence. Did you generate them?")
					raise
				X.append(sequence)
				y.append(self.get_class_one_hot(videos[1]))
		return np.array(X), np.array(y)


	def get_extracted_sequence(self,video):
		"""Get the saved extracted features."""
		filename = video[2]
		path = os.path.join(self.sequence_path, filename + '.npy')
		if os.path.isfile(path):
			return np.load(path)
		else:
			return None

