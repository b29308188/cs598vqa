from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Activation,  Dropout, Reshape
#from keras.layers.merge import concatenate
from keras.layers import Merge, Lambda
from keras.layers.merge import _Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam 
from keras.engine.topology import Layer
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



_sketch_op = tf.load_op_library('/save/lchen112/CLEVR_v1.0/src/CBP/build/count_sketch.so')
#@tf.RegisterGradient('CountSketch')
#def _count_sketch_grad(op, grad):
#    probs, h, s, _ = op.inputs
#    input_size = int(probs.get_shape()[1])
#    return [_sketch_op.count_sketch_grad(grad, h, s, input_size), None, None, None]

class MCB(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.img_dim = 2048
        self.word_vec_dim = 300
        super(MCB, self).__init__(**kwargs)
    def build(self, input_shape):
        h = self.add_weight(shape = (self.img_dim, ), initializer = tf.random_uniform_initializer(0, self.output_dim), trainable = False)
        self.h1 = tf.cast(h, 'int32')
        h = self.add_weight(shape = (self.img_dim, ), initializer = tf.random_uniform_initializer(0, self.output_dim), trainable = False)
        self.h2 = tf.cast(h, 'int32')
        s = self.add_weight(shape = (self.word_vec_dim, ), initializer = tf.random_uniform_initializer(0, 2), trainable = False)
        self.s1 = tf.cast(tf.floor(s) * 2 - 1, 'int32') # 1 or -1
        s = self.add_weight(shape = (self.word_vec_dim, ), initializer = tf.random_uniform_initializer(0, 2), trainable = False)
        self.s2 = tf.cast(tf.floor(s) * 2 - 1, 'int32') # 1 or -1
        super(MCB, self).build(input_shape)
    def count_sketch(self, probs, project_size, h, s):
        with tf.variable_scope('CountSketch_'+probs.name.replace(':', '_')) as scope:
            input_size = int(probs.get_shape()[1])
            # h, s must be sampled once
            history = tf.get_collection('__countsketch')
            if scope.name in history:
                scope.reuse_variables()
            tf.add_to_collection('__countsketch', scope.name)
        sk = _sketch_op.count_sketch(probs, h, s, project_size)
        sk.set_shape([probs.get_shape()[0], project_size])
        return sk
    def call(self, x):
        p1 = self.count_sketch(x[:, :img_dim], self.output_dim, self.h1, self.s1)
        p2 = self.count_sketch(x[:, img_dim:], self.output_dim, self.h2, self.s2)
        pc1 = tf.complex(p1, tf.zeros_like(p1))
        pc2 = tf.complex(p2, tf.zeros_like(p2))
        conved = tf.ifft(tf.fft(pc1) * tf.fft(pc2))
        return tf.real(conved)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

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
            model.add(MCB(output_dim = 16000, input_shape = (img_dim + word_vec_dim,)))
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
            model.add(Merge([image_model, language_model], mode='concat', concat_axis=1))
            model.add(MCB(output_dim = 16000, input_shape = (img_dim + word_vec_dim,)))
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
