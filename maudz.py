#!/usr/bin/python3

# This code mostly copied from Pieter, then tweaked.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import sklearn
from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
#from sklearn.preprocessing import LabelEncoder
import deeplift
from deeplift.conversion import keras_conversion as kc

#instantiate scaler
scaler = sklearn.preprocessing.MinMaxScaler()

fulldata  = pd.read_csv("data/fl_cn_training_data.csv")

one_meanz = [ 
    list(fulldata[fulldata['SL_RD_copy_estimate']==1][fulldata.columns.difference(['SL_RD_copy_estimate','Time'])].apply( lambda row: np.mean(row))) 
  ] 

meanz = [ 
    list(fulldata[fulldata['SL_RD_copy_estimate']==0][fulldata.columns.difference(['SL_RD_copy_estimate','Time'])].apply( lambda row: np.mean(row))),
    list(fulldata[fulldata['SL_RD_copy_estimate']==1][fulldata.columns.difference(['SL_RD_copy_estimate','Time'])].apply( lambda row: np.mean(row))) 
  ] 

zero_meanz = list(fulldata[fulldata['SL_RD_copy_estimate']==0][fulldata.columns.difference(['SL_RD_copy_estimate','Time'])].apply( lambda row: np.mean(row))) 

#for func in [np.mean,np.median,np.var]:
#  print(func)
#  print(pd.DataFrame(fulldata).apply(lambda row: func(row)))

train_data = sklearn.utils.shuffle(fulldata)

X = train_data[train_data.columns.difference(['SL_RD_copy_estimate','Time'])]
X = scaler.fit_transform(X.astype(float))
Y = train_data['SL_RD_copy_estimate'].values

amodel = keras.models.Sequential()
amodel.add(keras.layers.Dense(39,input_dim=13, activation='relu'))
amodel.add(keras.layers.Dense(9, input_dim=39, activation='relu'))
amodel.add(keras.layers.Dense(3, input_dim=9 , activation='relu'))
amodel.add(keras.layers.Dense(1, input_dim=3))
amodel.compile(loss='mean_squared_error', optimizer='adam')
amodel.fit(X, Y, epochs=3, batch_size=100)

### A lot of the below is just copied straight from the deeplift 
### example of how to use it

#NonlinearMxtsMode defines the method for computing importance scores.
#NonlinearMxtsMode.DeepLIFT_GenomicsDefault uses the RevealCancel rule on Dense layers
#and the Rescale rule on conv layers (see paper for rationale)
#Other supported values are:
#NonlinearMxtsMode.RevealCancel - DeepLIFT-RevealCancel at all layers (used for the MNIST example)
#NonlinearMxtsMode.Rescale - DeepLIFT-rescale at all layers
#NonlinearMxtsMode.Gradient - the 'multipliers' will be the same as the gradients
#NonlinearMxtsMode.GuidedBackprop - the 'multipliers' will be what you get from guided backprop
#Use deeplift.util.get_integrated_gradients_function to compute integrated gradients
#Feel free to email avanti [dot] shrikumar@gmail.com if anything is unclear

deeplift_model = kc.convert_sequential_model(
  amodel,
  nonlinear_mxts_mode=deeplift.blobs.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)

#Specify the index of the layer to compute the importance scores of.
#In the example below, we find scores for the input layer, which is idx 0 in deeplift_model.get_layers()
find_scores_layer_idx = 0

#Compile the function that computes the contribution scores
#For sigmoid or softmax outputs, target_layer_idx should be -2 (the default)
#(See "3.6 Choice of target layer" in https://arxiv.org/abs/1704.02685 for justification)
#For regression tasks with a linear output, target_layer_idx should be -1
#(which simply refers to the last layer)
#If you want the DeepLIFT multipliers instead of the contribution scores, you can use get_target_multipliers_func

deeplift_contribs_func = deeplift_model.get_target_contribs_func(
  find_scores_layer_idx=find_scores_layer_idx,
  target_layer_idx=-1)

#You can also provide an array of indices to find_scores_layer_idx to get scores for multiple layers at once

#compute scores on inputs
#input_data_list is a list containing the data for different input layers
#eg: for MNIST, there is one input layer with with dimensions 1 x 28 x 28
#In the example below, let X be an array with dimension n x 1 x 28 x 28 where n is the number of examples
#task_idx represents the index of the node in the output layer that we wish to compute scores.
#Eg: if the output is a 10-way softmax, and task_idx is 0, we will compute scores for the first softmax class

predictions = amodel.predict(X)

scorez = pd.DataFrame(deeplift_contribs_func(task_idx=0,
  input_data_list=[X],
  input_references_list=[zero_meanz],
  batch_size=100,
  progress_update=100000))

scorez.columns = train_data.columns[0:13]

scorez['real'] = Y
scorez['pred'] = predictions

scorez.to_csv('scores_trained_on_all.csv')


