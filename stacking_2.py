from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.cross_validation import KFold
import pandas as pd
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
#%%
data = pd.read_csv('/home/eugene/Santander/input/train.csv')
data_test = pd.read_csv('/home/eugene/Santander/input/test.csv')

labels = data.TARGET
data.drop('TARGET',1,inplace=True)

#remove ID
data.drop('ID',1,inplace=True)
data_test.drop('ID',1,inplace=True)
#%% ADD ID (it seems remove ID is bad idea xD) - no, it isn't
data = pd.read_csv('/home/eugene/prepr1.csv')
data_test = pd.read_csv('/home/eugene/prepr1_test.csv')

data_0 = pd.read_csv('/home/eugene/Santander/input/train.csv')
data_test_0 = pd.read_csv('/home/eugene/Santander/input/test.csv')
labels = data_0.TARGET
data_0.drop('TARGET',1,inplace=True)

data = pd.concat([data,data_0.ix[:,["ID"]]],axis = 1, join_axes=[data_0.index])
data_test = pd.concat([data_test,data_test_0.ix[:,["ID"]]],axis = 1, join_axes=[data_test_0.index])
del data_0,data_test_0
#%%
data_0 = pd.read_csv('/home/eugene/Santander/input/train.csv')
data_na = data
data_na_test = data_test
data = data.fillna(0)
data_test = data_test.fillna(0)
#%%data_2
data = pd.read_csv('/home/eugene/data_2.csv')
data_test = pd.read_csv('/home/eugene/data_test_2.csv')

data.drop('ID',1,inplace=True)
data_test.drop('ID',1,inplace=True)
#%%
clfs = [RandomForestClassifier(n_estimators=400, n_jobs=-1, criterion='gini',max_depth=20),
            RandomForestClassifier(n_estimators=400, n_jobs=-1, criterion='entropy',max_depth=20),
            ExtraTreesClassifier(n_estimators=400, n_jobs=-1, criterion='gini',max_depth=20),
            ExtraTreesClassifier(n_estimators=400, n_jobs=-1, criterion='entropy',max_depth=20),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=100),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=7, n_estimators=100),
            KNeighborsClassifier(n_neighbors=450,metric = 'chebyshev')]
#1-st stage predictions           
predictions_train_1st = np.zeros((data.shape[0], len(clfs)))
predictions_test_1st = np.zeros((data_test.shape[0], len(clfs)))
#%%
k_folds=5
kf = list(KFold(len(data), k_folds))

for i,clf in enumerate(clfs):
    print i,clf
    for j, (train, test) in enumerate(kf):
        print "Fold", j , len(test)
        #all data except k-th fold/ labels
        X=data.ix[train,]
        Z=labels.ix[train,]
        #k-th fold
        Y=data.ix[test,]
        #fit n-th classifier to data except k-th fold
        clf.fit(X, Z)
        #make predictions to k-th fold (k-th part of 1-st stage prediction)
        predictions_train_1st[test,i] = clf.predict_proba(Y)[:,1]
    #merge predictions for trainingset
    #make predictions for testset
    predictions_test_1st[:,i] = clf.predict_proba(data_test)[:,1]
del X,Y,Z
#%% prepare data for 2-nd stage classifier
data_2nd = pd.concat([data,pd.DataFrame(predictions_train_1st)],axis=1, join_axes=[data.index])
data_test_2nd = pd.concat([data_test,pd.DataFrame(predictions_test_1st)],axis=1, join_axes=[data_test.index])
#%% 2-nd stage classifier
#clf = LogisticRegression()
clf = xgb.XGBClassifier(n_estimators=110, nthread=-1, max_depth = 4,seed=1729)
#scores_xgb_noid = cross_validation.cross_val_score(clf,data,labels,cv=5,scoring = 'roc_auc')

clf.fit(data_2nd.drop('ID',1), labels)
submission = clf.predict_proba(data_test_2nd.drop('ID',1))[:,1]
submission = pd.DataFrame(submission)
#print "Linear stretch of predictions to [0,1]"
#submission = (submission - submission.min()) / (submission.max() - submission.min())

#subm = pd.DataFrame({"ID":data_test_0.ID, "TARGET": submission})
subm = pd.DataFrame({"ID":data_test_0.ID, "TARGET": submission.ix[:,0]})

subm.to_csv("submission.csv", index=False)

#%%
from sklearn import cross_validation

#clf = xgb.XGBClassifier(silent=False, nthread=4, max_depth=10, n_estimators=80, subsample=0.5, learning_rate=0.03, seed=42)
#n_estimators = 572 (550 seems better with lr 0.02)
clf = xgb.XGBClassifier(silent=False, nthread=4, max_depth=5, n_estimators=100, learning_rate=0.02, seed=1234)
scores = cross_validation.cross_val_score(clf,data_2nd,labels,cv=5,scoring = 'roc_auc')
print np.mean(scores)

#%%
import matplotlib.pyplot as plt

clf.fit(data_2nd, labels)
bst = clf.booster()
imps = bst.get_fscore()

#plt.bar(range(len(imps)), imps.values(), align="center")

a ={}
for k, v in imps.items() :
    if v > 50:
        a.update({k:v})
        

data_new = data_2nd.ix[:,a.keys()]

clf = xgb.XGBClassifier(silent=False, nthread=4, max_depth=5, n_estimators=550, learning_rate=0.02, seed=42)
scores = cross_validation.cross_val_score(clf,data_new,labels,cv=5,scoring = 'roc_auc')
#%%
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation

clf_svm = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=True)
neigh = KNeighborsClassifier(n_neighbors=5)
#svm scores 
scores = cross_validation.cross_val_score(clf_svm,data_new,labels,cv=2,scoring = 'roc_auc')
#knn scores
scores = cross_validation.cross_val_score(neigh,data_new,labels,cv=2,scoring = 'roc_auc')