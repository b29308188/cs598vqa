# cs598vqa 
This repository contains the experiment code on the CLEVR dataset.  
 
Prerequisite:	
	Python 2.7.6 
	Keras >= 2.0.2  
	Tensorflow >= 1.0.1  
	Scikit-learn >= 0.18.1  
	Spacy >= 1.7.3 
	h5py >= 2.6.0 
 
Source Files: 
	extract_features.py : Extract the iamge features from ResNet50 
	trainMLP: Train the Multiple Layer Perceptrons model on the training set 
	evaluateMLP: Evaluate the performance on the validation set 
 
Peroformances:	
	Accuracy: 47.75% (with Batch Normalization) ; 42.12 (with no Batch Normalization)  

