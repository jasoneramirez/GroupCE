###one-for-many counterfactuals

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import statistics
from numpy import linalg as LA
import math

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans

from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import random
import seaborn as sns
import matplotlib.pyplot as plt


# boston data
def datos():
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
    x['CHAS']=x['CHAS'].astype('category') #categorical feature
    y=boston['target']
    return x, y

#classication model
def modelo(x,y,tipo):
    random.seed(10)
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.33)
    #Logistic Regresion
    if tipo=='LR':
        model = LogisticRegression(solver='liblinear', random_state=0,C=10.0) 
        model.fit(x_train,y_train)

        y_pred_prob=model.predict_proba(x_test)
        y_pred=model.predict(x_test)

        w=model.coef_
        b=model.intercept_

        
    elif tipo=='SVM_linear':
        #SVM
        #model2=SVC()#
        #parameters =# {'kernel':('linear','rbf'), 'C':[1e0, 1e1, 1e2, 1e3],'gamma':np.logspace(-2, 2, 5)}
        #grid_sv = Gr#idSearchCV(model2, param_grid=parameters)
        #grid_sv.fit(#x_train, y_train)
        #print("Best classifier :", grid_sv.best_estimator_)
        model=SVC(kernel="linear",gamma=0.01,probability=True)
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        w=model.coef_
        b=model.intercept_
     
        
        
    return x_test,y_pred, model, w,b

def feature_type(dataset):
    features=list(dataset.columns)
    feat_type = ['Categorical' if x.name == 'category' else 'Numerical' for x in dataset.dtypes]
    features_type = dict(zip(features, feat_type))
    return features_type


#The data:
x2,y2=datos()
features_type=feature_type(x2)

### LINEAR MODEL


#The model:
x_test,y_pred, model, w,b=modelo(x2,y2,"LR")
y_pred_todos=model.predict(x2)

#Instance x0 
x0=x2.iloc[y_pred_todos==-1,]


# ONE-FOR-ALL


def oneforall_linear_hard(w,b,x0,features_type,phinu):

    features=list(x0.columns)
    indices=list(x0.index)
    n=x0.shape[0]

    m=gp.Model("oneforalllinear")

    #parameters
    bb=b
    ww={}
    for i in range(len(features)):
        ww[features[i]]=w[0][i]

    #Decision variables

    x_l={}
    for f in features:
        if  features_type[f]=='Numerical':
            x_l[f]=m.addVar(lb=0,ub=1,name='counterf_'+f) 
        elif features_type[f]=='Categorical':
            x_l[f]=m.addVar(vtype=GRB.BINARY,name='counterf_'+f) 

    #Objective

    m.setObjective(gp.quicksum((x0[f][i]-x_l[f])*(x0[f][i]-x_l[f]) for f in features for i in indices), GRB.MINIMIZE)

    #constraints f>=0

    m.addConstr(gp.quicksum(ww[f]*x_l[f] for f in features)+b>=phinu, "score")

    

    m.optimize()

    counterf={}
    for f in features:
        counterf[f]=x_l[f].getAttr(GRB.Attr.X)

    changes=[]
    for i in indices:
        change={}
        for f in features:
            change[f]=x_l[f].getAttr(GRB.Attr.X)-x0[f][i]
        changes.append(change)

    chang=pd.DataFrame(changes)

    counterf=pd.Series(counterf)

    return counterf, chang



counterf,chang=oneforall_linear_hard(w,b,x0,features_type,0)



# ONE-FOR-MANY

#number prototypes: P

P=3


def alternating_lineal_manual_hard(xsol,x0,features_type,P,w,b,kmax,phinu):


    #initial solution calculated like kmeans++ 
    prototypes = [xsol.sample().squeeze()] 
    for _ in range(P-1):
    # Calculate distances from points to the prototypes
        dists = np.sum([np.array([sum((prot-xsol.iloc[i])**2) for i in range(xsol.shape[0])]) for prot in prototypes],axis=0)
    # Standarize the distance to 0,1 and ensure they sum 1
        dists_s = (dists - np.min(dists))/(np.max(dists)-np.min(dists))
        dists_s /= np.sum(dists_s)
    # Choose remaining points based on their distances
        new_prot_idx, = np.random.choice(range(xsol.shape[0]), size=1, p=dists_s)
        prototypes += [xsol.iloc[new_prot_idx]]
   
    #prototypes=sol_ini
    k=0
    prev_prototypes=None
    while np.not_equal(prev_prototypes,prototypes).any() and k<kmax:
        prev_prototypes=prototypes
        clusters=[[] for _ in range(P)]
        for i in range(x0.shape[0]): 
            distances=[sum((prev_prototypes[k]-x0.iloc[i])**2) for k in range(P)] 
            prototype_index=np.argmin(distances) ¡
            clusters[prototype_index].append(x0.iloc[i])
        prototypes=[oneforall_linear_hard(w,b,pd.DataFrame(clusters[j]),features_type,phinu)[0] if clusters[j]!=[] else prev_prototypes[j] for j in range(P)]
        if all(x in clusters for x in [[], []]):
            print('k: '+str(k)+', collapsed')
        k+=1
    return prototypes, clusters, k



kmax=30
xpos=x2.iloc[y_pred_todos==1,] 
xsol=xpos
#phinu=0,-log(3/7),-log(1/9)
phinu=-(math.log(1/9))
P=3
prototypes, clusters,k=alternating_lineal_manual_hard(xsol,x0,features_type,P,w,b,kmax,phinu)

#save to then visualize:

with open('counterf_P3_nu09_LR2.json', 'w') as file:
     file.write(json.dumps(pd.DataFrame(prototypes).to_json())) 

with open('cluster1_P3_nu09_LR2.json', 'w') as file:
     file.write(json.dumps(pd.DataFrame(clusters[0]).to_json())) 
with open('cluster2_P3_nu09_LR2.json', 'w') as file:
     file.write(json.dumps(pd.DataFrame(clusters[1]).to_json())) 
with open('cluster3_P3_nu09_LR2.json', 'w') as file:
     file.write(json.dumps(pd.DataFrame(clusters[2]).to_json())) 


