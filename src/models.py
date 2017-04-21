from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Activation,  Dropout, Reshape
#from keras.layers.merge import concatenate
from keras.layers import Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM

from sklearn import preprocessing
from sklearn.externals import joblib

import numpy as np
import spacy

import tensorflow as tf
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth=True
sess = tf.Session(config=config_proto)

img_dim = 2048
word_vec_dim = 300
max_len = 50

class BOW_QI:
    def __init__(self):
        self.name = "../models/BOW_QI"	
    
    def build(self, num_classes, num_hiddens = 1024, num_layers = 3, dropout = 0.5, activation = "tanh"):
        print "Building the MLP..."
        model = Sequential()
        model.add(Dense(num_hiddens, input_dim=img_dim+word_vec_dim, init='uniform'))
        model.add(BatchNormalization())
	model.add(Dropout(dropout))
        for i in range(num_layers-1):
	    model.add(Dense(num_hiddens, init='uniform'))
	    model.add(Activation(activation))
	    model.add(Dropout(dropout))
        model.add(Dense(num_classes, init='uniform'))
        model.add(Activation('softmax'))
        json_string = model.to_json()
        open(self.name  + ".json", 'w').write(json_string)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ["accuracy"])
        self.model = model

    def extract_question_feature(self, nlp, q): 
        return nlp(q).vector*len(nlp(q))

    def train_on_batch(self, V, Q, A):
        return self.model.train_on_batch(np.hstack((V, Q)), A)

    def save_weights(self, suffix):
        self.model.save_weights(self.name + suffix + ".hdf5" )
    
    def load(self, suffix = "0"):
        self.model = model_from_json(open(self.name + ".json").read())
        self.model.load_weights(self.name + suffix + ".hdf5" )
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ["accuracy"])
    
    def predict(self, V, Q):
	return self.model.predict_classes(np.hstack((V, Q)), verbose=0)

class LSTM_QI:
    def __init__(self):
        self.name = "../models/LSTM_QI"	
    
    def build(self, num_classes, num_layers_lstm = 1, num_hiddens_lstm = 512,
            num_hiddens = 1024, num_layers = 3, dropout = 0.5, activation = "tanh"):
        print "Building the LSTM..."
	image_model = Sequential()
        image_model.add(Dense(num_hiddens, input_dim=img_dim, init='uniform'))
        image_model.add(BatchNormalization())
	#image_model.add(Reshape(input_shape = (img_dim,), dims=(img_dim,)))
	language_model = Sequential()
	if num_layers_lstm == 1:
	    language_model.add(LSTM(output_dim = num_hiddens_lstm, return_sequences=False, input_shape=(max_len, word_vec_dim)))
	else:
	    language_model.add(LSTM(output_dim = num_hiddens_lstm, return_sequences=True, input_shape=(max_len, word_vec_dim)))
	    for i in range(num_layers_lstm-2):
		language_model.add(LSTM(output_dim = num_hiddens_lstm, return_sequences=True))
	    language_model.add(LSTM(output_dim = num_hiddens_lstm, return_sequences=False))

	model = Sequential()
	model.add(Merge([image_model, language_model], mode='concat', concat_axis=1))
	#model.add(concatenate([language_model, image_model]))
        for i in range(num_layers):
	    model.add(Dense(num_hiddens, init='uniform'))
	    model.add(Activation(activation))
	    model.add(Dropout(dropout))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))
        json_string = model.to_json()
        open(self.name  + ".json", 'w').write(json_string)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ["accuracy"])
        self.model = model

    def extract_question_feature(self, nlp, q): 
	x = np.zeros((max_len, word_vec_dim))
	tokens = nlp(q)
	for i, token in enumerate(tokens):
	    if i >= max_len:
                print "Reach max_len"
                break
	    x[i,:] = tokens[i].vector
        return x

    def train_on_batch(self, V, Q, A):
        return self.model.train_on_batch([V, Q], A)

    def save_weights(self, suffix):
        self.model.save_weights(self.name + suffix + ".hdf5" )
    
    def load(self, suffix = "0"):
        self.model = model_from_json(open(self.name + ".json").read())
        self.model.load_weights(self.name + suffix + ".hdf5" )
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ["accuracy"])
    
    def predict(self, V, Q):
	return self.model.predict_classes([V, Q], verbose=0)
