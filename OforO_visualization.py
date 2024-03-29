
import numpy as np
import pandas as pd
import OforO_run
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import model_training 
import OforO_model_opt
from sklearn import preprocessing
from numpy import linalg as l2
import json
import pickle
import math



#example

#data and data characteristics

def feature_type(dataset):
        features=list(dataset.columns)
        feat_type = ['Categorical' if x.name == 'category' else 'Numerical' for x in dataset.dtypes]
        features_type = dict(zip(features, feat_type))
        return features_type

def datos():
    from sklearn.datasets import load_boston
    dataset=load_boston()
    dataset.keys()
    boston = pd.DataFrame(dataset.data, columns = dataset.feature_names)
    min_max_scaler = preprocessing.MinMaxScaler()
    boston_s= min_max_scaler.fit_transform(boston)
    boston_s = pd.DataFrame(boston_s,columns=boston.columns)
    boston=boston_s
    boston["MEDV"] = dataset.target
    boston['target']=boston['MEDV'].apply(lambda x: -1 if x<=22 else 1) #classification
    boston=boston.drop('MEDV', axis=1) 
    boston['target']=boston['target'].astype('category', copy=False) 
    x=boston.drop('target',axis=1)
    x['CHAS']=x['CHAS'].astype('category') #categorical features
    y=boston['target']
    return x, y

x,y=datos()
features=x.columns
features_type=feature_type(x)
index_cat=[i for (f,i) in zip(features,range(len(features))) if features_type[f]=='Categorical']
index_cont=[i for (f,i) in zip(features,range(len(features))) if features_type[f]=='Numerical']

#create a classification model

#RF
leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical,constraints_left_categorical, index, x, y, y_pred, y_pred_prob, model_clas, tree_data=model_training.randomforest(100,3,x,y)

#SVM or LR
x,y,y_pred,model,w, b=model_training.linear_model('LR',x,y)




#create the optimization model

#rf
objective="l2l0global"
model_opt_col=OforO_model_opt.modelo_opt_rf_nonseparable(leaves,index_cont,index_cat,objective,'False')
#or
model_opt_col=OforO_model_opt.modelo_opt_lineal_nonseparable(index_cont,index_cat,objective)




#define x0 and solve

x0=x[y_pred.squeeze()==-1][0:10]
y0=y_pred[y_pred.squeeze()==-1][0:10].squeeze()



indices=x0.index
perc=10 #value of |I*|
nu=0.5 #probability threshold
lam=0.1 #lam*l0+l2

#define timelimit
timelimit=10000




sol_ini={}
data=OforO_run.optimization_collective(x0,y0,perc,model_opt_col,leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical,constraints_left_categorical,index_cont,index_cat,model_clas,tree_data,timelimit,sol_ini,lam,nu)

#or
data=OforO_run.optimization_lineal_collective(x0,y0,perc,model_opt_col,w,b,index_cont,index_cat,timelimit,lam,nu)




#visualize the heatmap
plt.clf()
sns.set(rc={'figure.figsize':(12, 8)})
ax = sns.heatmap(data,linewidths=.5,center=0,vmin=-0.3,vmax=0.3, cmap="PiYG",yticklabels=False,square=True)
plt.yticks(rotation=0)
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
plt.savefig('l0globRFnu07_2.png')
#plt.savefig('RFinst6y11lam001_concondlip.png')
plt.show()

