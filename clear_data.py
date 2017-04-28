import itertools
import pandas as pd

data = pd.read_csv('/home/eugene/Santander/input/train.csv')
data_test = pd.read_csv('/home/eugene/Santander/input/test.csv')
labels = data.TARGET
data.drop('TARGET',1,inplace=True)

#pca_data = mdp.pca(data.as_matrix(),svd=True) 

#%% remove constant values
cols_to_remove = list()
for i in range(0,len(data.columns)):
    if len(set(data.ix[:,i])) == 1:
#        data.drop([i],inplace=True)
        cols_to_remove.append(data.columns[i])
        print(i," column to remove")
data.drop(cols_to_remove,axis=1,inplace=True)
data_test.drop(cols_to_remove,axis=1,inplace=True)
# remove identical features
col_comb = itertools.combinations(data.columns,2)
cols_to_remove = list()
for i,j in col_comb:
    feature_1 = i
    feature_2 = j
    if ((feature_1 not in cols_to_remove)and(feature_2 not in cols_to_remove)):
        if (list(data.ix[:,i]) == list(data.ix[:,j])):
            print(i,"and",j,"are equals")
            cols_to_remove.append(feature_2)
            
data.drop(cols_to_remove,axis=1,inplace=True)
data_test.drop(cols_to_remove,axis=1,inplace=True)
#write to scv file
data.to_csv("prepr0.csv", index=False)
data_test.to_csv("prepr0_test.csv", index=False)