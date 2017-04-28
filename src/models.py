from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Activation,  Dropout, Reshape
#from keras.layers.merge import concatenate
from keras.layers import Merge, Lambda
from keras.layers.merge import _Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam 
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

import keras.backend as K


class BOW_QI:
    def __init__(self, joint_method = "concat", lr = 0.0001):
        self.joint_method = joint_method
        self.lr = lr
        self.name = "../models/BOW" + self.joint_method + str(self.lr)	
    
    def build(self, num_classes, num_hiddens = 1024, num_layers = 3, dropout = 0.5, activation = "tanh"):
        print "Building the BOW..."
        model = Sequential()
        if self.joint_method == "concat":
            model.add(Dense(num_hiddens, input_dim=img_dim+word_vec_dim, init='uniform'))
        elif self.joint_method == "mcb":
            self.h1, self.s1, self.h2, self.s2 = None, None, None, None
            def mcb(x, project_dim = 16000, img_dim = 2048):
                from CBP import bilinear_pool
                return bilinear_pool(x[:, :img_dim], x[:, img_dim:], project_dim)
            model.add(Lambda(mcb, input_shape = (img_dim + word_vec_dim,)))
	    model.add(Dense(num_hiddens, init='uniform'))
        elif self.joint_method == "mul":
	    image_model = Sequential()
            image_model.add(Dense(num_hiddens, input_dim=img_dim, init='uniform'))
            #image_model.add(BatchNormalization())
            language_model = Sequential()
            language_model.add(Dense(num_hiddens, input_dim=word_vec_dim, init='uniform'))
            #language_model.add(BatchNormalization())
            model.add(Merge([image_model, language_model], mode='mul', concat_axis=1))
	    model.add(Dense(num_hiddens, init='uniform'))
        model.add(BatchNormalization())
	model.add(Activation(activation))
        #model.add(BatchNormalization())
        for i in range(num_layers-1):
	    model.add(Dense(num_hiddens, init='uniform'))
	    model.add(Activation(activation))
	    model.add(Dropout(dropout))
        model.add(Dense(num_classes, init='uniform'))
        model.add(Activation('softmax'))
    
        json_string = model.to_json()
        open(self.name  + ".json", 'w').write(json_string)
        model.compile(loss='categorical_crossentropy', optimizer= Adam(lr = self.lr), metrics = ["accuracy"])
        #model.compile(loss='categorical_crossentropy', optimizer= "adam", metrics = ["accuracy"])
        self.model = model

    def extract_question_feature(self, nlp, q): 
        return nlp(q).vector*len(nlp(q))

    def train_on_batch(self, V, Q, A):
        if self.joint_method !=  "mul":
            return self.model.train_on_batch(np.hstack((V, Q)), A)
        else:
            return self.model.train_on_batch([V, Q], A)

    def save_weights(self, suffix):
        self.model.save_weights(self.name + suffix + ".hdf5" )
    
    def load(self, suffix):
        self.model = model_from_json(open(self.name + ".json").read())
        self.model.load_weights(self.name + suffix + ".hdf5" )
        self.model.compile(loss='categorical_crossentropy', optimizer= Adam(lr = self.lr), metrics = ["accuracy"])
    
    def predict(self, V, Q):
        if self.joint_method !=  "mul":
	    return self.model.predict_classes(np.hstack((V, Q)), verbose=0)
        else:
	    return self.model.predict_classes([V, Q], verbose=0)

class LSTM_QI:
    def __init__(self, joint_method = "concat", lr = 0.001):
        self.joint_method = joint_method
        self.lr = lr
        self.name = "../models/LSTM" + self.joint_method + str(self.lr)	
    
    def build(self, num_classes, num_layers_lstm = 1, num_hiddens_lstm = 512,
            num_hiddens = 1024, num_layers = 3, dropout = 0.5, activation = "tanh"):
        print "Building the LSTM..."
	image_model = Sequential()
        image_model.add(Dense(num_hiddens, input_dim=img_dim, init='uniform'))
        image_model.add(BatchNormalization())
	language_model = Sequential()
	if num_layers_lstm == 1:
	    language_model.add(LSTM(output_dim = num_hiddens, return_sequences=False, input_shape=(max_len, word_vec_dim)))
	else:
	    language_model.add(LSTM(output_dim = num_hiddens_lstm, return_sequences=True, input_shape=(max_len, word_vec_dim)))
	    for i in range(num_layers_lstm-2):
		language_model.add(LSTM(output_dim = num_hiddens_lstm, return_sequences=True))
	    language_model.add(LSTM(output_dim = num_hiddens, return_sequences=False))
        language_model.add(BatchNormalization())
	
        model = Sequential()
        if self.joint_method == "concat":
	    model.add(Merge([image_model, language_model], mode='concat', concat_axis=1))
        elif self.joint_method == "mcb":
            def mcb(x, project_dim = 16000, img_dim = num_hiddens):
                from CBP import bilinear_pool
                return bilinear_pool(x[:, :img_dim], x[:, img_dim:], project_dim)
            model.add(Merge([image_model, language_model], mode='concat', concat_axis=1))
            model.add(Lambda(mcb, input_shape = (img_dim + word_vec_dim,)))
        elif self.joint_method == "mul":
            model.add(Merge([image_model, language_model], mode='mul', concat_axis=1))
	
        model.add(Dense(num_hiddens, init='uniform'))
        model.add(BatchNormalization())
	model.add(Activation(activation))
        for i in range(num_layers-1):
	    model.add(Dense(num_hiddens, init='uniform'))
	    model.add(Activation(activation))
	    model.add(Dropout(dropout))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))
        json_string = model.to_json()
        open(self.name  + ".json", 'w').write(json_string)
        model.compile(loss='categorical_crossentropy', optimizer= Adam(lr = self.lr), metrics = ["accuracy"])
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
        self.model.compile(loss='categorical_crossentropy', optimizer= Adam(lr = self.lr), metrics = ["accuracy"])
    
    def predict(self, V, Q):
	return self.model.predict_classes([V, Q], verbose=0)
