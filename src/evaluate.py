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
    parser.add_argument('-batch_size', type=int, default=128)
    args = parser.parse_args()
    
    print "Loading models..."
    labelencoder = joblib.load('../models/labelencoder.pkl')
    #model = BOW_QI()
    model = LSTM_QI()
    model.load("0")
        
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
        V, Q, A = [], [], []
        for i in range(args.batch_size*batch, args.batch_size*(batch+1)):
            if i >= len(questions_val):
                break
            question = questions_val[i]
            V.append(images_val[question["image_index"]])
            Q.append(model.extract_question_feature(nlp, question["question"]))
            try:
                A.append(labelencoder.transform([question["answer"]]))
            except:
                A.append(0)
        V = np.vstack(V)
        if len(Q[0].shape) == 1:
            Q = np.vstack(Q)
        else:
            Q = np.dstack(Q)
        A = np.vstack(A)
        predA = model.predict(V, Q)
        for a, pa in zip(A, predA):
            total += 1
            if a == pa:
                hit += 1
	progbar.add(args.batch_size, values=[("Accuracy", hit/total)])
    print "Accuracy", hit / total
