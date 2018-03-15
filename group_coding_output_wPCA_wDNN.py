from __future__ import print_function

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

wt_df = np.loadtxt("WT_FL1.csv", delimiter=",", skiprows=1)
one_df = np.loadtxt("OneCopy_FL1.csv", delimiter=",", skiprows=1)
two_df = np.loadtxt("TwoCopy_FL1.csv", delimiter=",", skiprows=1)

print(wt_df.shape)

print("Neg:")

print(np.mean(wt_df, axis=0))

print("OneCopy:")

print(np.mean(one_df, axis=0))

print("TwoCopy:")

print(np.mean(two_df, axis=0))

#
# _ = plt.plot(wt_df[:,0], wt_df[:,2], marker=".", linestyle="none", color="blue", alpha=0.1)
# _ = plt.plot(one_df[:,0], one_df[:,2], marker=".", linestyle="none", color="red", alpha=0.1)
# _ = plt.plot(two_df[:,0], two_df[:,2], marker=".", linestyle="none", color="green", alpha=0.1)
# _ = plt.xlabel("Forward Scatter")
# _ = plt.ylabel("FL1")
# _ = plt.margins(0.02)
# _ = plt.yscale("log")
# _ = plt.xscale("log")
# plt.show()

wt_corr = np.corrcoef(wt_df[:,0], wt_df[:,2])[0,1]
one_corr = np.corrcoef(one_df[:,0], one_df[:,2])[0,1]
two_corr = np.corrcoef(two_df[:,0], two_df[:,2])[0,1]

print(wt_corr, one_corr, two_corr)

def ecdf(data):

    x=np.sort(data)
    y=np.arange(1,len(x)+1)/len(x)

    return x, y

# wt_df[:,2] = wt_df[:,2]/wt_df[:,0]
# one_df[:,2] = one_df[:,2]/one_df[:,0]
# two_df[:,2] = two_df[:,2]/two_df[:,0]
#
# wt_x, wt_y = ecdf(wt_df[:,2])
# one_x, one_y = ecdf(one_df[:,2])
# two_x, two_y = ecdf(two_df[:,2])
#
# _ = plt.plot(wt_x, wt_y, marker=".", linestyle="none", color="blue", alpha=0.1)
# _ = plt.plot(one_x, one_y, marker=".", linestyle="none", color="red", alpha=0.1)
# _ = plt.plot(two_x, two_y, marker=".", linestyle="none", color="green", alpha=0.1)
# _ = plt.xlabel("Normalized FL1")
# _ = plt.ylabel("ECDF")
# _ = plt.margins(0.02)
# _ = plt.xlim(0,0.5)
#
# plt.show()

def ecdf_plot(list_of_data, colors):
    for i in range(len(list_of_data)):
        x, y = ecdf(list_of_data[i])
        _ = plt.plot(x, y, marker=".", linestyle="none", color=colors[i], alpha=0.1)

    _ = plt.margins(0.02)
    plt.show()

#ecdf_plot([wt_df[:,2],one_df[:,2],wt_df[:,2]], ["blue", "red", "green"])

def bootstrapper(x, func, size=1):
    bs_reps = np.empty(size)

    for i in range(size):
        bs_reps[i] = func(np.random.choice(x, size=len(x), replace=True))

    return bs_reps

def perm_boots(data1, data2, func, size=1):
    perm_reps = np.empty(size)
    all_data = np.concatenate((data1, data2))

    for i in range(size):
        reshuffle_data = np.random.permutation(all_data)
        data1_random = reshuffle_data[:len(data1)]
        data2_random = reshuffle_data[len(data1):]

        perm_reps[i] = func(data1_random) - func(data2_random)

    return perm_reps

# one_bs = bootstrapper(one_df[:,2], np.mean, size=1000)
#
# _ = plt.hist(one_bs, bins=50, normed=True)
# plt.show()

fl_wt_mean = np.mean(wt_df[:,2])
fl_one_mean = np.mean(one_df[:,2])
fl_two_mean = np.mean(two_df[:,2])

bs_one_wt = perm_boots(wt_df[:10,2], one_df[:10,2], np.mean, size=2000)

_ = plt.hist(bs_one_wt, bins=50, normed=True)
plt.show()

p_val = np.sum(bs_one_wt >= (fl_one_mean - fl_wt_mean)) / len(bs_one_wt)

print("CIs:")
print(np.percentile(bs_one_wt, [2.5, 97.5]))

print("P-value:")
print(p_val)


bs_two_one = perm_boots(one_df[:,2], two_df[:,2], np.mean, size=10)

_ = plt.hist(bs_two_one, bins=50, normed=True)
plt.show()

p_val = np.sum(bs_two_one >= (fl_two_mean - fl_one_mean)) / len(bs_two_one)

print("Means:")
print(fl_wt_mean, fl_one_mean, fl_two_mean)

print("CIs:")
print(np.percentile(bs_two_one, [2.5, 97.5]))

print("P-value:")
print(p_val)

### PCA Section
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

pca = PCA()
scaler = MinMaxScaler()

wt_df_pd = pd.read_csv("C:/Users/pieter/WT_FL1.csv")
one_df_pd = pd.read_csv("C:/Users/pieter/OneCopy_FL1.csv")
two_df_pd = pd.read_csv("C:/Users/pieter/TwoCopy_FL1.csv")

wt_df_pd['copy_number']=0
one_df_pd['copy_number']=1
two_df_pd['copy_number']=2

union_df = one_df_pd.append(two_df_pd)
union_df = union_df.append(wt_df_pd)

union_df_drop = union_df.drop(['copy_number'],axis=1)

df_scaled = pd.DataFrame(scaler.fit_transform(union_df_drop), columns=union_df_drop.columns)
pca.fit(df_scaled)
features = range(pca.n_components_)

plt.bar(features,pca.explained_variance_)

union_1=union_df.loc[union_df['copy_number'] == 1]
union_2=union_df.loc[union_df['copy_number'] == 2]

plt.scatter(union_1['FSC.A'],union_1['FL1.A'],c='r',alpha=0.1)
plt.scatter(union_2['FSC.A'],union_2['FL1.A'],c='b',alpha=0.1)
plt.show()


### A  DNN classifier
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

#instantiate scaler
scaler = MinMaxScaler()

#read in data
wt_df_pd = pd.read_csv("WT_FL1.csv")
one_df_pd = pd.read_csv("OneCopy_FL1.csv")
two_df_pd = pd.read_csv("TwoCopy_FL1.csv")

#add category type
wt_df_pd['copy_number']=0
one_df_pd['copy_number']=1
two_df_pd['copy_number']=2

#append dfs
union_df = one_df_pd.append(two_df_pd)
union_df = union_df.append(wt_df_pd)

#shuffle row
union_df=shuffle(union_df)

#export values as np.ndarray
union_dataset = union_df.values

# load scalar transformed dataset and copy_number category
X = scaler.fit_transform(union_dataset[:,0:3].astype(float))

# copy_number category converted into disctinct columns 
encoder = LabelEncoder()
encoder.fit(union_dataset[:,3])
encoded_Y = encoder.transform(union_dataset[:,3])

# one hot encode on those categorical columns
Y = np_utils.to_categorical(encoded_Y)

# define model
def threeway_model():
    model = Sequential()
    model.add(Dense(27, input_dim=3, activation='relu'))
    model.add(Dense(9, input_dim=3, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# evaluate performance using validation and kfold cross-validation
estimator = KerasClassifier(build_fn=threeway_model, validation_split=0.33, epochs=3, verbose=1)
kfold = KFold(n_splits=3, shuffle=True)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# detailed evaluation of predictions
estimator.fit(X, Y, validation_split=0.33, epochs=3, verbose=1)
predict = estimator.predict(X)

# initiate counts, using ints for readability
miss_ct = 0
is0_not1 = 0
is0_not2 = 0
is1_not0 = 0
is1_not2 = 0
is2_not0 = 0
is2_not1 = 0

for index in range(len(X)):
    if predict[index] != union_dataset[:,3][index]:
        #print(predict[index],union_dataset[:,3],X[index])
        miss_ct+=1
        if predict[index] == 1 and union_dataset[:,3][index] == 0:
            is0_not1 += 1
        if predict[index] == 2 and union_dataset[:,3][index] == 0:
            is0_not2 += 1
            
        if predict[index] == 0 and union_dataset[:,3][index] == 1:
            is1_not0 += 1
        if predict[index] == 2 and union_dataset[:,3][index] == 1:
            is1_not2 += 1
            
        if predict[index] == 0 and union_dataset[:,3][index] == 2:
            is2_not0 += 1
        if predict[index] == 1 and union_dataset[:,3][index] == 2:
            is2_not1 += 1

# output counts  
print('Total miss categorizations: ',str(miss_ct),'\n')          
outline = ('is Zero, not One: {} ({}%), not Two: {} ({}%)\n').format(is0_not1,int(100*is0_not1/miss_ct),is0_not2,int(100*is0_not2/miss_ct))
print(outline)
outline = ('is One, not Zero: {} ({}%), not Two: {} ({}%)\n').format(is1_not0,int(100*is1_not0/miss_ct),is1_not2,int(100*is1_not2/miss_ct))
print(outline)
outline = ('is Two, not Zero: {} ({}%), not One: {} ({}%)\n').format(is2_not0,int(100*is2_not0/miss_ct),is2_not1,int(100*is2_not1/miss_ct))
print(outline)
