import sys
import argparse
import json
import h5py
from random import shuffle
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
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
    parser.add_argument('-num_hidden_units', type=int, default=1024)
    parser.add_argument('-num_hidden_layers', type=int, default=3)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-activation', type=str, default='tanh')
    parser.add_argument('-num_epochs', type=int, default=100)
    parser.add_argument('-model_save_interval', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=128)
    args = parser.parse_args()
    
    print "Loading questions..."
    questions_train = json.load(open("../questions/CLEVR_train_questions.json", "r"))["questions"]
    print "Loading image features..."
    images_train = h5py.File("./features.h5")["train"][:]
    print "Loading word vectors"
    nlp = spacy.load("en", add_vectors = lambda vocab : vocab.load_vectors(open("/save/lchen112/glove.42B.300d.txt","r")))
    #encode labels
    labelencoder = preprocessing.LabelEncoder()
    labelencoder.fit([q["answer"] for q in questions_train])
    nb_classes = len(list(labelencoder.classes_))
    joblib.dump(labelencoder,'../models/labelencoder.pkl')
	
    img_dim = 2048
    word_vec_dim = 300
    
    model = Sequential()
    print "Building the model..."
    model.add(Dense(args.num_hidden_units, input_dim=img_dim+word_vec_dim, init='uniform'))
    model.add(BatchNormalization())
    if args.dropout>0:
	model.add(Dropout(args.dropout))
    for i in range(args.num_hidden_layers-1):
	model.add(Dense(args.num_hidden_units, init='uniform'))
	model.add(Activation(args.activation))
        if args.dropout>0:
	    model.add(Dropout(args.dropout))
    model.add(Dense(nb_classes, init='uniform'))
    model.add(Activation('softmax'))
    json_string = model.to_json()
    model_file_name = '../models/mlp_batchnorm_num_hidden_units_' + str(args.num_hidden_units) + '_num_hidden_layers_' + str(args.num_hidden_layers)	
    open(model_file_name  + '.json', 'w').write(json_string)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    #questions_train = questions_train[:128]
    print 'Training...'
    for epoch in xrange(args.num_epochs):
	print "epoch", epoch
        index_shuf = range(len(questions_train))
	shuffle(index_shuf)
        progbar = generic_utils.Progbar(len(questions_train))
        for batch in range(int(np.ceil(len(questions_train)/args.batch_size))):
            X_batch = []
            Y_batch = []
            for i in range(args.batch_size*batch, args.batch_size*(batch+1)):
                if i >= len(questions_train):
                    break
                q = questions_train[index_shuf[i]]
                x = np.hstack((images_train[q["image_index"]], nlp(q["question"]).vector*len(nlp(q["question"]))))
                X_batch.append(x)
                y = labelencoder.transform([q["answer"]])
                y = np_utils.to_categorical(y, nb_classes)
                Y_batch.append(y)
            X_batch = np.vstack(X_batch)
            Y_batch = np.vstack(Y_batch)
	    #print model.predict_classes(X_batch, verbose=0)
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(args.batch_size, values=[("train loss", loss)])
	if epoch % args.model_save_interval == 0:
            model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(epoch))
    model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(epoch))
