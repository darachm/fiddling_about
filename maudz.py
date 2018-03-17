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

wt_df_pd  = pd.read_csv("data/WT_FL1.csv")
one_df_pd = pd.read_csv("data/OneCopy_FL1.csv")
two_df_pd = pd.read_csv("data/TwoCopy_FL1.csv")

meanz = [
    list(wt_df_pd.apply( lambda row: np.mean(row))),
    list(one_df_pd.apply(lambda row: np.mean(row))),
    list(two_df_pd.apply(lambda row: np.mean(row)))
  ]

#add category type
wt_df_pd['copy_number']  = 0
one_df_pd['copy_number'] = 1
two_df_pd['copy_number'] = 2

union_df = one_df_pd.append(two_df_pd)
union_df = union_df.append(wt_df_pd)
union_df = sklearn.utils.shuffle(union_df)
union_dataset = union_df.values

X = scaler.fit_transform(union_dataset[:,0:3].astype(float))

for func in [np.mean,np.median,np.var]:
  print(func)
  print(pd.DataFrame(union_dataset).apply(lambda row: func(row)))

encoder = sklearn.preprocessing.LabelEncoder()
encoder.fit(union_dataset[:,3])
encoded_Y = encoder.transform(union_dataset[:,3])

# one hot encode on those categorical columns
Y = keras.utils.np_utils.to_categorical(encoded_Y)

def threeway_model():
  model = keras.models.Sequential()
#  model.add(keras.layers.Dense(27, input_dim=3, activation='relu'))
  model.add(keras.layers.Dense(1, input_dim=3, activation='relu'))
  model.add(keras.layers.Dense(3, activation='softmax'))
  # Compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return(model)

# evaluate performance using validation and kfold cross-validation
#estimator = KerasClassifier(build_fn=threeway_model,validation_split=0.33, epochs=3, verbose=1)
#kfold = KFold(n_splits=3, shuffle=True)
#results = cross_val_score(estimator, X, Y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# detailed evaluation of predictions
#estimator.fit(X, Y, validation_split=0.33, epochs=3, verbose=1)
#predict = estimator.predict(X)

# refit so I have a real
amodel = threeway_model()
amodel.fit(X, Y, epochs=3, batch_size=100)

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
  target_layer_idx=-2)

#You can also provide an array of indices to find_scores_layer_idx to get scores for multiple layers at once

#compute scores on inputs
#input_data_list is a list containing the data for different input layers
#eg: for MNIST, there is one input layer with with dimensions 1 x 28 x 28
#In the example below, let X be an array with dimension n x 1 x 28 x 28 where n is the number of examples
#task_idx represents the index of the node in the output layer that we wish to compute scores.
#Eg: if the output is a 10-way softmax, and task_idx is 0, we will compute scores for the first softmax class

predictions = amodel.predict(X)
which_predicted = [i.argsort()[-1] for i in predictions]

output = None

for real,pred in [ (x,y) for x in [0,1,2] for y in[0,1,2] ]:
#  print("Looking for contributions of variables to predicting a"+real+" as a "+pred" , relative to wild-type means.")
  tmp_X = X[ [i==pred and j==real  for i,j in zip(which_predicted,encoded_Y)  ]  ]
  tmp_result = pd.DataFrame(deeplift_contribs_func(task_idx=pred,
    input_data_list=[tmp_X],
    input_references_list=meanz[real],
    batch_size=100,
    progress_update=10000))
  tmp_result['real'] = real
  tmp_result['pred'] = pred 
  try:
    output = output.append(tmp_result)
  except:
    output = tmp_result

output = output.rename(columns={0:'FSCmaybe',1:'SSCmaybe',2:'FL1maybe'})

output.to_csv('scores_by_predictions.csv')


#scores1 = np.array(deeplift_contribs_func(task_idx=1,
#  input_data_list=[X],
#  input_references_list=wt_meanz,
#  batch_size=100,
#  progress_update=10000))
#np.savetxt("scores1.csv", scores1, delimiter=",")

