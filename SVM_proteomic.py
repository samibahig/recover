import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
#inline
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import sklearn
import os
plt.style.use('ggplot')
#print('\nMETADATA :')
data_path = ''
metadata_filename = data_path + 'metadata.csv'
meta_df = pd.read_csv(metadata_filename)
#print(meta_df.columns)
meta_df.columns = ['#', 'plate', '-', 'symptoms'] + list(meta_df)[4:]
#print(meta_df.columns)
#print('available metadata :', list(meta_df))
meta_idx = meta_df['ID'].to_list()
meta_label = meta_df['symptoms'].to_list()
#print('------------------')
#print(list(zip(meta_idx, meta_label)))
#print('------------------')
meta_id_label_dict = {str(k): 1 if v=='S' else 0 for k, v in zip(meta_idx, meta_label)}
data_path = ''
#import pdb; pdb.set_trace()
#DF1 : proteomics
#print('\nPROEOMICS DATA :')
proteomics_data_filename = 'proteomics.csv'

dim_df = pd.read_csv(proteomics_data_filename, nrows=1)
#print('--------')
#print(dim_df)
#print('--------')
dim = len(list(dim_df))
#print(dim)
#print('------------')
print('# of columns in source csv file :', dim)
all_cols = [i for i in range(dim)]

print('--------')
feat_cols = all_cols[1:-4]
print(feat_cols)
print('--------')
samplesidx_col = [0]

feat_df = pd.read_csv(proteomics_data_filename, skiprows=4, nrows=1, dtype=str, usecols=feat_cols)
features = list(feat_df)
print('# of features : ', len(features))
print('first feature :', features[0])
print('last feature :', features[-1])

idx_df = pd.read_csv(proteomics_data_filename, skiprows=6, index_col=0, skipfooter=4, usecols=[0], engine='python')
idx = list(idx_df.index.values)
print('# of idx : ', len(idx))
print('first id :', idx[0])
print('last id :', idx[-1])

df1 = pd.read_csv(proteomics_data_filename, skiprows=6, dtype=np.float32, skipfooter=4, usecols=feat_cols, engine='python')
assert df1.shape[0] == len(idx)
assert df1.shape[1] == len(features)

df1['idx'] = idx
df1.set_index('idx', inplace=True)
df1.columns = features
print('# of Nan values :', df1.isna().sum().sum())

#clean data of samples that are not in metadata :
idx = df1.index.values
y = []
for k in range(len(idx)):
    id = idx[k]
    if id in meta_id_label_dict:
        y.append(meta_id_label_dict[id])
    else:
        # we will not put this sample in the dataset
        #print('sample to remove because of unknown label:', k, id)
        y.append('to_remove')
df1['label'] = y
df1 = df1[df1.label != 'to_remove']

#create X and y matrices for ML :
y = list(df1['label'])
del df1['label']
print('---------')
print(df1)
print('---------')
X = df1.to_numpy()
print('proteomics data :')
print('# of samples : ', df1.shape[0])
print('# of features : ', df1.shape[1])
print('labels:', list(dict.fromkeys(y)))


## save X and y in pickles if you want :
##data_name = 'recover_multiomics_'
##feat_dict = {k: str(v) for k, v in zip(range(len(list(df))), list(df))}
##with open(data_path + data_name + 'feat_dict', 'wb') as fo:
##    pkl.dump(feat_dict, fo)
##with open(data_path + data_name + 'X', 'wb') as fo:df
##            pkl.dump(X, fo)
##with open(data_path + data_name + 'y', 'wb') as fo:
##            pkl.dump(y, fo)


from sklearn import svm
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_test = y_test[0]
import numpy as np


clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(predictions, y_test))