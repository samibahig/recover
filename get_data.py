"""
This script extract a data and metadata from the files
it gives dataframes and a labels file
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

data_path = ''

print('\nMETADATA :')
metadata_filename = data_path + 'metadata.csv'
meta_df = pd.read_csv(metadata_filename)
meta_df.columns = ['#', 'plate', '-', 'symptoms'] + list(meta_df)[4:]
print('available metadata :', list(meta_df))
meta_idx = meta_df['ID'].to_list()
meta_label = meta_df['symptoms'].to_list()
meta_id_label_dict = {str(k): 1 if v=='S' else 0 for k, v in zip(meta_idx, meta_label)}

#DF1 : proteomics
print('\nPROTEOMICS DATA :')
proteomics_data_filename = data_path + 'proteomics.csv'

dim_df = pd.read_csv(proteomics_data_filename, nrows=1)
dim = len(list(dim_df))
print('# of columns in source csv file :', dim)
all_cols = [i for i in range(dim)]
feat_cols = all_cols[1:-4]
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
X = df1.to_numpy()
print('proteomics data :')
print('# of samples : ', df1.shape[0])
print('# of features : ', df1.shape[1])
print('labels:', list(dict.fromkeys(y)))

# DF 2 : proteomics (cytokines)
print('\nPROTEOMICS CYTOKINES DATA :')
proteomics_cyt_data_filename = data_path + 'proteomics_cyt.csv'

dim_df = pd.read_csv(proteomics_cyt_data_filename, nrows=1)
dim = len(list(dim_df))
print('# of columns in source csv file :', dim)
all_cols = [i for i in range(dim)]
feat_cols = all_cols[1:-2]
samplesidx_col = [0]

feat_df = pd.read_csv(proteomics_cyt_data_filename, skiprows=4, nrows=1, dtype=str, usecols=feat_cols)
features = list(feat_df)
print('# of features : ', len(features))
print('first feature :', features[0])
print('last feature :', features[-1])

idx_df = pd.read_csv(proteomics_cyt_data_filename, skiprows=7, index_col=0, skipfooter=14, usecols=[0], engine='python')
idx = list(idx_df.index.values)
print('# of idx : ', len(idx))
print('first id :', idx[0])
print('last id :', idx[-1])

df2 = pd.read_csv(proteomics_cyt_data_filename, skiprows=7, dtype=np.float32, skipfooter=14, usecols=feat_cols, na_values=['> ULOQ'], engine='python')
assert df2.shape[0] == len(idx)
assert df2.shape[1] == len(features)

df2['idx'] = idx
df2.set_index('idx', inplace=True)
df2.columns = features
print('# of Nan values :', df2.isna().sum().sum())

#clean data of samples that are not in metadata :
idx = df2.index.values
y = []
for k in range(len(idx)):
    id = idx[k]
    if id in meta_id_label_dict:
        y.append(meta_id_label_dict[id])
    else:
        # we will not put this sample in the dataset
        #print('sample to remove because of unknown label:', k, id)
        y.append('to_remove')
df2['label'] = y
df2 = df2[df2.label != 'to_remove']

#create X and y matrices for ML :
y = list(df2['label'])
del df2['label']
X = df2.to_numpy()
print('proteomics_cyt data :')
print('# of samples : ', df2.shape[0])
print('# of features : ', df2.shape[1])
print('labels:', list(dict.fromkeys(y)))

## an option to manage Nan values :
##replace nans with column mean :
##print('replacing nans with column mean')
#for col in list(df2):
#    #print(int(df[col].mean()))
#    df2[col].fillna(int(df2[col].mean()), inplace=True)
#
##print('is there NaN values ? :', df2.isnull().values.any())
##print('# of Nan values :', df2.isna().sum().sum())

## to concatenate the 2 proteomics dataframes if you want :
##df_1_2 = pd.concat([df1, df2], axis=1)

print('\nMETABOLOMICS DATA :')
metabolomics_data_filename = data_path + 'metabolomics.csv'
feat_df = pd.read_csv(metabolomics_data_filename, index_col=0, skiprows=[0], dtype=str, usecols=[0])
features = list(feat_df.index.values)
print('# of features : ', len(features))
print('first feature :', features[0])
print('last feature :', features[-1])

idx_df = pd.read_csv(metabolomics_data_filename, header=1, nrows=1)
idx = list(idx_df)[1:]
idx = [l[17:22] for l in idx]
print('# of idx : ', len(idx))

labels_df = pd.read_csv(metabolomics_data_filename, nrows=1)
labels = list(labels_df)[1:]
print('# of labels : ', len(labels))

cols_df = pd.read_csv(metabolomics_data_filename, header=1, nrows=1)
cols_list = list(cols_df)

df3 = pd.read_csv(metabolomics_data_filename, header=1, dtype=np.float32, na_values=['#DIV/0!'], usecols=cols_list[1:])
df3 = df3.T
df3['idx'] = idx
df3.set_index('idx', inplace=True)
df3.columns = features
df3 = df3.dropna(axis=1)

#clean data of samples that are not in metadata :
idx = df3.index.values
y = []
for k in range(len(idx)):
    id = idx[k]
    if id in meta_id_label_dict:
        y.append(meta_id_label_dict[id])
    else:
        # we will not put this sample in the dataset
        #print('sample to remove because of unknown label:', k, id)
        y.append('to_remove')
df3['label'] = y
df3 = df3[df3.label != 'to_remove']

#create X and y matrices for ML :
y = list(df3['label'])
del df3['label']
X = df3.to_numpy()

print('metabolomics data :')
print('# of samples : ', df3.shape[0])
print('# of features : ', df3.shape[1])
print('labels:', list(dict.fromkeys(y)))

## to concatenate the 2 proteomics dataframes and the metabolomics if you want :
##df = pd.concat([df_1_2, df3], axis=1)
##df = df.dropna(axis=0)
##print('multi-omics df :')
##print('# of samples : ', df.shape[0])
##print('# of features : ', df.shape[1])

## save X and y in pickles if you want :
##data_name = 'recover_multiomics_'
##feat_dict = {k: str(v) for k, v in zip(range(len(list(df))), list(df))}
##with open(data_path + data_name + 'feat_dict', 'wb') as fo:
##    pkl.dump(feat_dict, fo)
##with open(data_path + data_name + 'X', 'wb') as fo:
##            pkl.dump(X, fo)
##with open(data_path + data_name + 'y', 'wb') as fo:
##            pkl.dump(y, fo)


