import glob
import numpy as np
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
import tensorflow as tf
import h5py
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth=True
sess = tf.Session(config=config_proto)
base_model = ResNet50(weights = "imagenet")
model = Model(input = base_model.input, output = base_model.get_layer("flatten_1").output)
image_prefix = "../images"
batch_size = 256
h5f = h5py.File("./features.h5", "w")
for split in ["train", "val", "test"]:
    image_dir = image_prefix + "/" + split + "/"
    xs = []
    image_paths = sorted(glob.glob(image_dir+"*.png"))
    M = []
    for i, img_path in enumerate(image_paths):
        if i % 5000 == 0:
            print "Processed", i
        img = image.load_img(img_path, target_size = (224, 224))
        x = image.img_to_array(img)
        xs.append(np.expand_dims(x, axis = 0))
        if i > 0 and (i % batch_size == 0 or i == len(image_paths) -1):
            X = np.concatenate(xs)
            X = preprocess_input(X)
            features = model.predict(X)
            M.append(features)
            xs = []
    M = np.concatenate(M)
    h5f.create_dataset(split, data = M)
h5f.close()
