import sys
import argparse
import json
import h5py
from random import shuffle
import numpy as np

from keras.models import model_from_json
from keras.utils import np_utils, generic_utils

from sklearn import preprocessing
from sklearn.externals import joblib

import spacy

import tensorflow as tf
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth=True
sess = tf.Session(config=config_proto)

if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-weights', type=str, default="../models/mlp_batchnorm_num_hidden_units_1024_num_hidden_layers_3_epoch_30.hdf5") 
    parser.add_argument('-model', type=str, default="../models/mlp_batchnorm_num_hidden_units_1024_num_hidden_layers_3.json")
    parser.add_argument('-batch_size', type=int, default=128)
    args = parser.parse_args()
    
    print "Loading models..."
    model = model_from_json(open(args.model).read())
    model.load_weights(args.weights)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    labelencoder = joblib.load('../models/labelencoder.pkl')
    
    print "Loading questions..."
    #questions_val = json.load(open("../questions/CLEVR_val_questions.json", "r"))["questions"]
    print "Loading image features..."
    #images_val = h5py.File("./features.h5")["val"][:]
    print "Loading word vectors"
    #nlp = spacy.load("en", add_vectors = lambda vocab : vocab.load_vectors(open("/save/lchen112/glove.42B.300d.txt","r")))
	
    print 'Evaluating...'
    total = 0.0
    hit = 0.0
    progbar = generic_utils.Progbar(len(questions_val))
    for batch in range(int(np.ceil(len(questions_val)/args.batch_size))):
        X_batch = []
        Y_batch = []
        for i in range(args.batch_size*batch, args.batch_size*(batch+1)):
            if i >= len(questions_val):
                break
            q = questions_val[i]
            x = np.hstack((images_val[q["image_index"]], nlp(q["question"]).vector*len(nlp(q["question"]))))
            X_batch.append(x)
            y = labelencoder.transform([q["answer"]])
            Y_batch.append(y)
        X_batch = np.vstack(X_batch)
        Y_batch = np.hstack(Y_batch)
	Y_predict = model.predict_classes(X_batch, verbose=0)
        for y, py in zip(Y_batch, Y_predict):
            total += 1
            if py == y:
                hit += 1
	progbar.add(args.batch_size, values=[("Accuracy", hit/total)])
    print "Accuracy", hit / total
