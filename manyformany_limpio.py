###many-for-many counterfactuals

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


#datos boston
def datos():
    dataset=load_boston()
    dataset.keys()
    boston = pd.DataFrame(dataset.data, columns = dataset.feature_names)
    min_max_scaler = preprocessing.MinMaxScaler()
    boston_s= min_max_scaler.fit_transform(boston)
    boston_s = pd.DataFrame(boston_s,columns=boston.columns)
    boston=boston_s
    boston["MEDV"] = dataset.target
    boston['target']=boston['MEDV'].apply(lambda x: -1 if x<=22 else 1) #lo paso a clasificacion
    boston=boston.drop('MEDV', axis=1) 
    boston['target']=boston['target'].astype('category', copy=False) 
    x=boston.drop('target',axis=1)
    x['CHAS']=x['CHAS'].astype('category') #no olvidar indicar cuales son categoricas
    y=boston['target']
    return x, y

def modelo(x,y,tipo):
    random.seed(10)
    #Boston Housing
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.33)
    #Logistic Regresion
    if tipo=='LR':
        model = LogisticRegression(solver='liblinear', random_state=0,C=10.0) #C regularizacion
        model.fit(x_train,y_train)

        y_pred_prob=model.predict_proba(x_test)
        y_pred=model.predict(x_test)

        w=model.coef_
        b=model.intercept_

        gamm=[] 
        labels=[]
        alphas=[]
        xsupport=[]
        
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
        gamm=[] 
        labels=[]
        alphas=[]
        xsupport=[]

    elif tipo=='SVM_RBF':
        model=SVC(kernel="rbf",probability=True)
        parameters={'C':[1e0, 1e1, 1e2, 1e3],'gamma':np.logspace(-2, 2, 5)}
        grid_sv=GridSearchCV(model,param_grid=parameters)
        grid_sv.fit(x_train,y_train)
        model.fit(x_train,y_train)
        model=grid_sv.best_estimator_
        y_pred=model.predict(x_test)
        gamm=model.gamma
        w=[]
        b=[]
        labels = np.sign(model.dual_coef_)
        alphas=abs(model.dual_coef_)
        xsupport=model.support_vectors_

        #Ns=model.dual_coef_.shape[1]
        
        
        
    return x_test,y_pred, model, w,b, gamm, labels, alphas,xsupport

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
x_test,y_pred, model, w,b, gamm, labels, alphas,xsupport=modelo(x2,y2,"LR")
y_pred_todos=model.predict(x2)

#Instance x0 and lam
x0=x2.iloc[y_pred_todos==-1,]


centroid=np.mean(x0,axis=0)
aux1=sum([np.sum((x0.iloc[i]-centroid)**2)for i in range(x0.shape[0])])
xpos=x2.iloc[y_pred_todos==1,] 
aux2=max([sum(w[0]*xpos.iloc[k])*x0.shape[0] for k in range(xpos.shape[0])])

lam=0.8

import pickle

with open('model_LR_bueno2.pkl', 'wb') as handle:
    pickle.dump(model, handle)

#with open('model_LR_bueno.pkl', 'rb') as handle:
#    model2=pickle.load(handle)

#w=model.coef_
#b=model.intercept_

# ONE-FOR MANY

def oneformany_linear(w,b,lam,x0,features_type):

    features=list(x0.columns)
    indices=list(x0.index)
    n=x0.shape[0]

    m=gp.Model("oneformanylinear")

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

    #to balance the two objectives
    #aux1=sum((x0[f][i]-xi[f])*(x0[f][i]-xi[f]) for f in features for i in indices)
    #aux2=sum(ww[f]*xi[f] for f in features)*n

    m.setObjective(lam/aux1*gp.quicksum((x0[f][i]-x_l[f])*(x0[f][i]-x_l[f]) for f in features for i in indices)-(1-lam)*(gp.quicksum(ww[f]*x_l[f] for f in features))*n/aux2, GRB.MINIMIZE)

    #constraints for the dummies?

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



def oneformany_linear_hard(w,b,x0,features_type,phinu):

    features=list(x0.columns)
    indices=list(x0.index)
    n=x0.shape[0]

    m=gp.Model("oneformanylinear")

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

    #to balance the two objectives
    #aux1=sum((x0[f][i]-xi[f])*(x0[f][i]-xi[f]) for f in features for i in indices)
    #aux2=sum(ww[f]*xi[f] for f in features)*n

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


counterf,chang=oneformany_linear(w,b,lam,x0,features_type)

counterf,chang=oneformany_linear_hard(w,b,x0,features_type,0)

model.predict_proba(counterf.values.reshape(1,-1))


import json

#with open('counterf_P1_lam06_todos_SVMLinear_c.txt', 'w') as file:
#     file.write(json.dumps(counterf.to_json())) 

#with open('camb_P1_lam06_todos_SVMLinear_c.txt', 'w') as file:
#     file.write(json.dumps(chang.to_json())) 


# MANY-FOR-MANY

#number prototypes: P

P=3




def alternating_lineal_manual(xsol,x0,features_type,P,lam,w,b,kmax):

    constant=((1-lam)*aux1*w)/(2*lam*aux2)
    x0_t=x0.apply(lambda x: x+constant[0],axis=1)

    #initial solution calculated like kmeans++ (trying with the positive data)
    prototypes = [xsol.sample().squeeze()] 
    for _ in range(P-1):
    # Calculate distances from points to the prototypes
        dists = np.sum([np.array([lam/aux1*sum((prot-xsol.iloc[i])**2)-(1-lam)*sum(w[0]*prot)/aux2  for i in range(xsol.shape[0])]) for prot in prototypes],axis=0)
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
        for i in range(x0.shape[0]): #para cada instancia calcula la distancia a los prototipos
            distances=[lam/aux1*sum((prev_prototypes[k]-x0.iloc[i])**2)-(1-lam)*sum(w[0]*prev_prototypes[k])/aux2 for k in range(P)] 
            prototype_index=np.argmin(distances) #miro a que prototipo (cluster) se une cada uno
            clusters[prototype_index].append(x0.iloc[i])
        prototypes=[oneformany_linear(w,b,lam,pd.DataFrame(clusters[j]),features_type)[0] if clusters[j]!=[] else prev_prototypes[j] for j in range(P)]
        if all(x in clusters for x in [[], []]):
            print('k: '+str(k)+', collapsed')
        k+=1
    return prototypes, clusters, k






def alternating_lineal_manual_hard(xsol,x0,features_type,P,w,b,kmax,phinu):


    #initial solution calculated like kmeans++ (trying with the positive data)
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
        for i in range(x0.shape[0]): #para cada instancia calcula la distancia a los prototipos
            distances=[sum((prev_prototypes[k]-x0.iloc[i])**2) for k in range(P)] 
            prototype_index=np.argmin(distances) #miro a que prototipo (cluster) se une cada uno
            clusters[prototype_index].append(x0.iloc[i])
        prototypes=[oneformany_linear_hard(w,b,pd.DataFrame(clusters[j]),features_type,phinu)[0] if clusters[j]!=[] else prev_prototypes[j] for j in range(P)]
        if all(x in clusters for x in [[], []]):
            print('k: '+str(k)+', collapsed')
        k+=1
    return prototypes, clusters, k




def alternating_lineal(w,b,lam,x0,P):

    constant=((1-lam)*aux1*w)/(2*lam*aux2)
    x0_t=x0.apply(lambda x: x+constant[0],axis=1)

    #clustering with kmeans
    kmeans = KMeans(n_clusters=P).fit(x0_t)   ###y las categoricas
    clusters=[x0.iloc[kmeans.labels_==j,] for j in range(P)]
    prototypes=[pd.Series(kmeans.cluster_centers_[j],index=x0.columns) for j in range(P)]
    return prototypes, clusters

lam=0.75
prototypes, clusters=alternating_lineal(w,b,lam,x0,P)


model.predict_proba(prototypes[2].values.reshape(1,-1))

kmax=30
xpos=x2.iloc[y_pred_todos==1,] 
y_prob_pos=np.array([p[1] for p in model.predict_proba(xpos)])
#xpos_frontera=xpos[y_prob_pos<=0.9] #positive class but no more than 70%
xsol=xpos
prototypes_m,clusters_m,k_m=alternating_lineal_manual(xsol,x0,features_type,P,lam,w,b,kmax)

kmax=30
xpos=x2.iloc[y_pred_todos==1,] 
xsol=xpos
#phinu=0,-log(3/7),-log(1/9)
phinu=-(math.log(1/9))
P=3
prototypes, clusters,k=alternating_lineal_manual_hard(xsol,x0,features_type,P,w,b,kmax,phinu)

model.predict_proba(prototypes[2].values.reshape(1,-1))


with open('counterf_P3_nu09_LR2.json', 'w') as file:
     file.write(json.dumps(pd.DataFrame(prototypes).to_json())) 

with open('cluster1_P3_nu09_LR2.json', 'w') as file:
     file.write(json.dumps(pd.DataFrame(clusters[0]).to_json())) 
with open('cluster2_P3_nu09_LR2.json', 'w') as file:
     file.write(json.dumps(pd.DataFrame(clusters[1]).to_json())) 
with open('cluster3_P3_nu09_LR2.json', 'w') as file:
     file.write(json.dumps(pd.DataFrame(clusters[2]).to_json())) 



### MODELO RADIAL

x_test,y_pred, model, w,b, gamm, labels, alphas,xsupport=modelo(x2,y2,"SVM_RBF")

#with open('model_SVMRadial_it.pkl', 'wb') as handle:
#    pickle.dump(model, handle)


L=gamm #Lipschitz constant
y_pred_todos=model.predict(x2)

#Instance x0 and lam
x0=x2.iloc[y_pred_todos==-1,]
#lam=0.6


# ONE-FOR-MANY 

xpos=x2.iloc[y_pred_todos==1,] 
y_prob_pos=np.array([p[1] for p in model.predict_proba(xpos)])


centroid=np.mean(x0,axis=0)
aux1=sum([np.sum((x0.iloc[i]-centroid)**2)for i in range(x0.shape[0])])
xpos=x2.iloc[y_pred_todos==1,] 
aux2=max([x0.shape[0]*sum(alphas[0][j]*labels[0][j]*math.exp(-gamm*LA.norm(xpos.iloc[k]-xsupport[j])**2) for j in range(xsupport.shape[0])) for k in range(xpos.shape[0])])
#aux2=640 ##valor del score mas alto en la base de datos



def obj_function(x,x0,lam,alphas,xsupport,gamm,labels):
    n=x0.shape[0]
    return lam*sum([np.sum((x0.iloc[i]-x)**2)for i in range(x0.shape[0])])/aux1-(1-lam)*n*sum(alphas[0][j]*labels[0][j]*math.exp(-gamm*LA.norm(x-xsupport[j])**2) for j in range(xsupport.shape[0]))/aux2



def obj_function_ind(prot,x0,lam,alphas,xsupport,gamm,labels,i):
    n=x0.shape[0]
    f=lam*np.sum((x0.iloc[i]-prot)**2)/aux1-(1-lam)*sum(alphas[0][j]*labels[0][j]*math.exp(-gamm*LA.norm(prot-xsupport[j])**2) for j in range(xsupport.shape[0]))/aux2
    return f

def value_c(x0,lam,L,fobj0,fobj,alphas,labels,gamm):

    #max (z-xi)=M+M^0
    #M^0=max|xi|
   
    M0=max(LA.norm(x0,axis=1)) 
    
    #value of M 
    n=x0.shape[0]
    ax=1/(2*lam*n)
    bx=2*lam*M0*n-lam*L*n+L*n
    cx=(-2*lam*M0*n+lam*L*n-L*n)**2
    dx=4*lam*n*(lam*M0**2*n+lam*fobj0*n-fobj0*n-fobj)
    #print(n)
    #print(fobj0)
    #print(fobj)
    #print(M0)
    #print(cx)
    #print(dx)


    M=ax*(bx+math.sqrt(cx-dx))
    #M=ax*(bx+math.sqrt(cx))

    maxdi2=(M+M0)**2
    

    mu=-sum(alphas[labels==1])*2*gamm+sum(alphas[labels==-1])*(2*gamm*math.exp(-gamm*maxdi2)-4*gamm**2*maxdi2)



    c=lam*n/aux1-n*(1-lam)/aux2*mu/2 
    

    return c

def h(x,c,features):
    return c*sum(x[f]*x[f] for f in features)


def subgrad_g(x,x0,lam,c,alphas,xsupport,gamm,labels):
    n=x0.shape[0]
    Ns=xsupport.shape[0] 
    return 2*c*x-2*lam*sum((x-x0.iloc[p]) for p in range(n))/aux1-2*n/aux2*(1-lam)*gamm*sum(alphas[0][i]*labels[0][i]*(x-xsupport[i])*math.exp(-gamm*LA.norm(x-xsupport[i])**2) for i in range(Ns))



def argmin(ynew,c,x0,features_type):

    #parameters
    features=list(x0.columns)

    #Decision variables

    m=gp.Model("batchcountrbf")

    x={}
    for f in features:
        if  features_type[f]=='Numerical':
            x[f]=m.addVar(lb=0,ub=1,name='counterf_'+f) #poner aqui upper and lower de cada variable
        elif features_type[f]=='Categorical':
            x[f]=m.addVar(vtype=GRB.BINARY,name='counterf_'+f) 


    #Objective
    m.setObjective( h(x,c,features)-sum(ynew[f]*x[f] for f in features), GRB.MINIMIZE)

    
    m.optimize()

    sol={}
    for f in features:
        sol[f]=x[f].getAttr(GRB.Attr.X)

    return pd.Series(sol)


def algoritmo_dca(x_i,x0,features_type,lam,alphas,xsupport,gamm,labels,tol,kmax):
    n=x0.shape[0]
    features=list(x0.columns)
    fobj0=obj_function(np.zeros(x0.shape[1]),x0,lam,alphas,xsupport,gamm,labels)
    fobj_i=obj_function(x_i,x0,lam,alphas,xsupport,gamm,labels)
    c=value_c(x0,lam,L,fobj0,fobj_i,alphas,labels,gamm)
    xold=x_i
    xnew=xold
    ynew=subgrad_g(xnew,x0,lam,c,alphas,xsupport,gamm,labels)
    xold=argmin(ynew,c,x0,features_type)
    k=1
    while (LA.norm(xnew-xold)>tol) and k<kmax:
        xnew=xold
        ynew=subgrad_g(xnew,x0,lam,c,alphas,xsupport,gamm,labels)
        xold=argmin(ynew,c,x0,features_type)
        k+=1

    return k,xold


#initial solution: SVM linear
x_i=counterf

#other parameters
tol= 1e-4
kmax=30
lam=0.6
k,counterf2=algoritmo_dca(x_i,x0,features_type,lam,alphas,xsupport,gamm,labels,tol,kmax)
print(k)
model.predict_proba(counterf2.values.reshape(1,-1))

#with open('counterf_P1_lam06_todos_SVMRadial_c.txt', 'w') as file:
#     file.write(json.dumps(counterf2.to_json())) 


# MANY-FOR-MANY







def alternating_radial(sol_ini,xsol,y_pred_todos,x0,features_type,P,lam,alphas,xsupport,gamm,labels,kmax,kmax2):

    total_objective=[]
    #prototypes=sol_ini #initial solution: linear SVM solution

    #initial solution calculated like kmeans++ (trying with the positive data)
    prototypes = [xsol.sample().squeeze()] 
    for _ in range(P-1):
    # Calculate distances from points to the closest prototype already chosen
        #dists = np.sum([np.array([obj_function_ind(prot,xsol,lam,alphas,xsupport,gamm,labels,i)for i in range(xsol.shape[0])]) for prot in prototypes],axis=0)
        dists= np.array([min([obj_function_ind(prot,xsol,lam,alphas,xsupport,gamm,labels,i)for prot in prototypes]) for i in range(xsol.shape[0])])
    # Standarize the distance to 0,1 and ensure they sum 1
        dists_s = (dists - np.min(dists))/(np.max(dists)-np.min(dists))
        dists_s /= np.sum(dists_s)
    # Choose remaining points based on their distances
        new_prot_idx, = np.random.choice(range(xsol.shape[0]), size=1, p=dists_s)
        prototypes += [xsol.iloc[new_prot_idx]]
    k2=0
    prev_prototypes=None
    while np.not_equal(prev_prototypes,prototypes).any() and k2<kmax:
        prev_prototypes=prototypes
        clusters=[[] for _ in range(P)]
        for i in range(x0.shape[0]): #para cada instancia calcula la distancia a los prototipos
            distances=[obj_function_ind(prev_prototypes[k],x0,lam,alphas,xsupport,gamm,labels,i) for k in range(P)] 
            prototype_index=np.argmin(distances) #miro a que prototipo (cluster) se une cada uno
            clusters[prototype_index].append(x0.iloc[i])
        if k2==0:
            total_objective.append(sum([obj_function(prev_prototypes[j],pd.DataFrame(clusters[j]),lam,alphas,xsupport,gamm,labels) for j in range(P)]))
        prototypes=[algoritmo_dca(prev_prototypes[j],pd.DataFrame(clusters[j]),features_type,lam,alphas,xsupport,gamm,labels,tol,kmax2)[1] if clusters[j]!=[] else prev_prototypes[j] for j in range(P)]
        if all(x in clusters for x in [[], []]):
            print('k2: '+str(k2)+', collapsed')
        k2+=1

        total_objective.append(sum([obj_function(prototypes[j],pd.DataFrame(clusters[j]),lam,alphas,xsupport,gamm,labels) for j in range(P)]))
    return prototypes, clusters, k2, total_objective


P=3
lam=0.6
kmax=20 #10
kmax2=10 #5
xpos_frontera=xpos[y_prob_pos<=90] #positive class but no more than 70%
xsol=xpos_frontera
#xsol=x0 # the negatives ones
#sol_ini=prototypes_m
sol_ini=[]



for it in range(10):
    prototypes3, clusters3, k2, total_objective = alternating_radial(sol_ini,xsol,y_pred_todos,x0,features_type,P,lam,alphas,xsupport,gamm,labels,kmax,kmax2)

    with open('counterf_P3_lam06_todos_SVMRadial_solinikmeans_it_'+str(it)+'_ver2.json', 'w') as file:
         file.write(json.dumps(pd.DataFrame(prototypes3).to_json())) 

    with open('totalobject_P3_lam06_todos_SVMRadial_solinikmeans_it_'+str(it)+'_ver2.json', 'w') as file:
         file.write(json.dumps(total_objective)) 

    with open('cluster1_P3_lam06_todos_SVMRadial_solinikmeans_it_'+str(it)+'_ver2.json', 'w') as file:
         file.write(json.dumps(pd.DataFrame(clusters3[0]).to_json())) 
    with open('cluster2_P3_lam06_todos_SVMRadial_solinikmeans_it_'+str(it)+'_ver2.json', 'w') as file:
         file.write(json.dumps(pd.DataFrame(clusters3[1]).to_json())) 
    with open('cluster3_P3_lam06_todos_SVMRadial_solinikmeans_it_'+str(it)+'_ver2.json', 'w') as file:
         file.write(json.dumps(pd.DataFrame(clusters3[2]).to_json())) 



#total_objective=[0.509, 0.508, 0.507, 0.506, 0.505, 0.503, 0.502, 0.501, 0.5, 0.499]

prototypes2
clusters2
model.predict_proba(prototypes3[0].values.reshape(1,-1))

clusters2[2]


#prueba
prototypes3, clusters3, k2, total_objective = alternating_radial(sol_ini,xsol,y_pred_todos,x0,features_type,P,lam,alphas,xsupport,gamm,labels,kmax,kmax2)