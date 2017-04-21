import argparse
import json
import h5py
from random import shuffle
import numpy as np
from keras.utils import np_utils, generic_utils
from sklearn import preprocessing
from sklearn.externals import joblib
import spacy
import tensorflow as tf
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth=True
sess = tf.Session(config=config_proto)

import models
reload(models)
from models import *

if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_epochs', type=int, default=2)
    parser.add_argument('-model_save_interval', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=128)
    args = parser.parse_args()
    
    print "Loading questions..."
    #questions_train = json.load(open("../questions/CLEVR_train_questions.json", "r"))["questions"]
    print "Loading image features..."
    #images_train = h5py.File("./features.h5")["train"][:]
    print "Loading word vectors"
    #nlp = spacy.load("en", add_vectors = lambda vocab : vocab.load_vectors(open("/save/lchen112/glove.42B.300d.txt","r")))
    #encode labels
    labelencoder = preprocessing.LabelEncoder()
    labelencoder.fit(sorted([q["answer"] for q in questions_train]))
    num_classes = len(list(labelencoder.classes_))
    joblib.dump(labelencoder,'../models/labelencoder.pkl')
	
    #model = BOW_QI()
    model = LSTM_QI()
    model.build(num_classes)
    print 'Training...'
    for epoch in xrange(args.num_epochs):
	print "epoch", epoch
        index_shuf = range(len(questions_train))
	shuffle(index_shuf)
        progbar = generic_utils.Progbar(len(questions_train))
        for batch in range(int(np.ceil(len(questions_train)/args.batch_size))):
            V, Q, A = [], [], []
            for i in range(args.batch_size*batch, args.batch_size*(batch+1)):
                if i >= len(questions_train):
                    break
                question = questions_train[index_shuf[i]]
                V.append(images_train[question["image_index"]])
                Q.append(model.extract_question_feature(nlp, question["question"]))
                A.append(np_utils.to_categorical(labelencoder.transform([question["answer"]]), num_classes))
            V = np.vstack(V)
            if len(Q[0].shape) == 1:
                Q = np.vstack(Q)
            else:
                Q = np.rollaxis(np.dstack(Q), -1)
            A = np.vstack(A)
	    #print model.predict_classes(X_batch, verbose=0)
            accuracy = model.train_on_batch(V, Q, A)[1]
            progbar.add(args.batch_size, values=[("Accuracy", accuracy)])
	if epoch % args.model_save_interval == 0:
            model.save_weights(str(epoch))
    model.save_weights(str(epoch))
