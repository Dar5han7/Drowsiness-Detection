import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Dropout,ZeroPadding3D,LSTM
from tensorflow.keras.models import Sequential,load_model

from collections import deque
import sys

class ResearchModels():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=2048):

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        #if self.nb_classes >= 10:
        #   metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        print("Loading LSTM model.")
        self.input_shape = (seq_length, features_length)
        self.model = self.lstm()
        # Now compile the network.
        optimizer = tf.keras.optimizers.Adam()
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)
	
        # print(self.model.summary())

    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""

        model = Sequential()
        model.add(LSTM(units=60, return_sequences=True, dropout=0.5, input_shape=self.input_shape))
        model.add(Flatten())
        model.add(Dense(units=20, activation='relu'))
        model.add(Dense(units=100, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

	
        return model

    
