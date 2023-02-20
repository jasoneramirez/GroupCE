
import numpy as np
import pandas as pd
import run_rf
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import model_training 
import modelo_opt
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
    boston['target']=boston['MEDV'].apply(lambda x: -1 if x<=22 else 1) #lo paso a clasificacion
    boston=boston.drop('MEDV', axis=1) 
    boston['target']=boston['target'].astype('category', copy=False) 
    x=boston.drop('target',axis=1)
    x['CHAS']=x['CHAS'].astype('category') #no olvidar indicar cuales son categoricas
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


#to reproduce the experiments of the paper

#import pickle
#with open('rf_norm_100_prune3_todos.pkl', 'rb') as f: 
#        leaves, values, restricciones_right_numerical, restricciones_left_numerical, restricciones_right_categorical,restricciones_left_categorical, index, x2, y, y_pred, y_pred_prob,model_clas,tree_data =pickle.load(f)

#constraints_right_categorical=[(a,b,3,d) for (a,b,c,d) in restricciones_right_categorical]
#constraints_left_categorical=[(a,b,3,d) for (a,b,c,d) in restricciones_left_categorical]
#constraints_right_numerical=[(a,b,c,d) if c<=2 else (a,b,c+1,d) for (a,b,c,d) in restricciones_right_numerical]
#constraints_left_numerical=[(a,b,c,d) if c<=2 else (a,b,c+1,d) for (a,b,c,d) in restricciones_left_numerical]



#create the optimization model
objective='l0l2'
model_opt=modelo_opt.modelo_opt_rf_separable(leaves,index_cont,index_cat,objective)

#collective
objective="l2l0global"
model_opt_col=modelo_opt.modelo_opt_rf_nonseparable(leaves,index_cont,index_cat,objective,'False')
#model_opt_col=modelo_opt_rf_nonseparable(leaves,index_cont,index_cat,objective,'False')

#lineal
objective='l0l2'
model_opt=modelo_opt.modelo_opt_lineal_separable(index_cont,index_cat,objective)

#linealcollective
objective="l2l0global"
model_opt_col=modelo_opt.modelo_opt_lineal_nonseparable(index_cont,index_cat,objective)

#model_opt_col=modelo_opt_lineal_nonseparable(index_cont,index_cat,objective)


#define x0 and solve

index_criminalidad_alta=x[x.CRIM>0.002812].index.tolist()
x_crim=x.loc[index_criminalidad_alta]
y_crim=y_pred.loc[index_criminalidad_alta]

x_pos=x_crim[y_crim.squeeze()==1][0:10].index.tolist()
x_neg=x_crim[y_crim.squeeze()==-1][0:10].index.tolist()

#separable
x0=x.iloc[x_pos+x_neg]
y0=y_pred.iloc[x_pos+x_neg].squeeze()
indices=x0.index
lam=0.01 #lambda*l0+l2

##voy a buscar dos instancias cercanas que tengan contrafácticos distintos
x0=x[y_pred.squeeze()==-1]
y0=y_pred[y_pred.squeeze()==-1].squeeze()
x01=x0.iloc[0]
distance=0.4
similares=[]
for i in range(x0.shape[0]):
    d=l2.norm(x01-x0.iloc[i])
    if d<=distance:
        #distance=d
        #x_sim=x0.iloc[i] #initial solution
        similares.append(x0.iloc[i].name)
#x0=x0.iloc[x0.index.isin([6,11])]
#y0=y0.iloc[y0.index.isin([6,11])]
x0=x0.iloc[x0.index.isin(similares)]
y0=y0.iloc[y0.index.isin(similares)]

x0=x0.iloc[x0.index.isin([6,7])]
y0=y0.iloc[y0.index.isin([6,7])]

x0=x0[0:10]
y0=y0[0:10]
indices=x0.index
lam=0.01
perc=2

#noseparable

x0=x[y_pred.squeeze()==-1]
y0=y_pred[y_pred.squeeze()==-1].squeeze()
indices=x0.index
perc=10 #number of people to be changed

#cojo los 10 primeros de la clase negativa 
x0=x[y_pred.squeeze()==-1][1:11]
y0=y_pred[y_pred.squeeze()==-1][1:11].squeeze()

#x0=x[y_pred.squeeze()==-1][0:10]
#y0=y_pred[y_pred.squeeze()==-1][0:10].squeeze()


indices=x0.index
perc=10
nu=0.5
lam=0.1

#define timelimit
timelimit=10000



result={}
final_class={}
x_sols={}
for i in indices:
        x_sols[i],result[i],final_class[i]=run_rf.optimization_individual(i,timelimit,x0,y0,lam,model_opt,leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical,constraints_left_categorical,index_cont,index_cat,model_clas,tree_data,x,y)


#lineal
for i in indices:
        x_sols[i],result[i],final_class[i]=run_rf.optimization_lineal_ind(i,x0,y0,lam,model_opt,w,b,index_cont,index_cat)



#no-separable

import pickle
with open('sol_ini_c.pkl', 'rb') as f:
        sol_ini=pickle.load(f)

sol_ini={}
data,fobj=run_rf.optimization_collective(x0,y0,perc,model_opt_col,leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical,constraints_left_categorical,index_cont,index_cat,model_clas,tree_data,timelimit,sol_ini,lam,nu)

data=run_rf.optimization_lineal_collective(x0,y0,perc,model_opt_col,w,b,index_cont,index_cat,timelimit,lam,nu)
data,fobj=optimization_collective(x0,y0,perc,model_opt_col,leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical,constraints_left_categorical,index_cont,index_cat,model_clas,tree_data,timelimit,sol_ini,lam,nu)


datas=[]
fobjs=[]
for nu in [0.8,0.9]:
    data,fobj=run_rf.optimization_collective(x0,y0,perc,model_opt_col,leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical,constraints_left_categorical,index_cont,index_cat,model_clas,tree_data,timelimit,sol_ini,lam,nu)
    datas.append(data)
    fobjs.append(fobj)

    with open('fobj'+str(nu)+'RF.txt', 'w') as file:
             file.write(json.dumps(fobj)) 
    with open('data'+str(nu)+'RF.pkl', 'wb') as handle:
        pickle.dump(data, handle)

with open('fobjsRF.pkl', 'wb') as handle:
        pickle.dump(fobjs, handle)
with open('datasRF.pkl', 'wb') as handle:
        pickle.dump(datas, handle)




datas=[]
fobjs=[]
for nu in [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]:
    #data,fobj=run_rf.optimization_lineal_collective(x0,y0,perc,model_opt_col,w,b,index_cont,index_cat,timelimit,lam,nu)
    data,fobj=optimization_lineal_collective(x0,y0,perc,model_opt_col,w,b,index_cont,index_cat,timelimit,lam,nu)
    datas.append(data)
    fobjs.append(fobj)

    with open('fobj'+str(nu)+'LR.txt', 'w') as file:
             file.write(json.dumps(fobj)) 
    with open('data'+str(nu)+'LR.pkl', 'wb') as handle:
        pickle.dump(data, handle)


with open('fobjs'+str(nu)+'LR.pkl', 'wb') as handle:
        pickle.dump(fobjs, handle)


nu=0.7
with open('data'+str(nu)+'RF.pkl','rb') as handle:
    data = pickle.load(handle)


# the heatmaps

def data_heatmap(resul,final_class,features,indices):

   

    resul_neg={} #ind que pasan de neg a pos
    resul_pos={} #ind que pasan de pos a neg


    for i in indices:
        if final_class[i]==1:
            resul_neg[i]=resul[i]
        elif final_class[i]==-1:
            resul_pos[i]=resul[i]


    data=pd.DataFrame(resul,index=features)

    data1=pd.DataFrame(resul_neg,index=features).transpose()
    data2=pd.DataFrame(resul_pos,index=features).transpose()


    return data1,data2

data1,data2=data_heatmap(result,final_class,features,indices)

plt.clf()
sns.set(rc={'figure.figsize':(12, 8)})
ax = sns.heatmap(data,linewidths=.5,center=0,vmin=-0.3,vmax=0.3, cmap="PiYG",yticklabels=False,square=True)
plt.yticks(rotation=0)
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
plt.savefig('l0globRFnu07_2.png')
#plt.savefig('RFinst6y11lam001_concondlip.png')
plt.show()

nus=[0.5,0.6,0.7,0.8,0.9]
fobjs=[0.7602444703087115,0.8802463725096771,0.9453772849199624,1.4369037028060685,1.796702315723751]

plt.clf()
plt.scatter(nus,fobjs)
plt.xlabel('Value of $\\nu$')
plt.ylabel('Objective function')
fig=plt.gcf()
fig.set_size_inches(9, 6)
plt.savefig('ParetoRFglob.png')
plt.show()

#sol=data.apply(pd.to_numeric).add(x0.apply(pd.to_numeric).reset_index(drop=True), fill_value=0)
#l2.norm(sol.iloc[0]-sol.iloc[1])

#con instancias 6 y 12
#con condicion distancia: dist(contrafacticos)=0.29
#sin condicion distancia: dist(contrafacticos)=1.05 (porque cambia una categorica)

#con instancias 6 y 10
#con condicion distancia: dist(contrafacticos)=0.37 
#sin condicion distancia: dist(contrafacticos)=0.45 


contrafacticos=x0.apply(pd.to_numeric).reset_index(drop=True).add(data.apply(pd.to_numeric),fill_value=0)

l2.norm(x0.iloc[0]-x0.iloc[1])
#instancias 6 y 7
#distancia: 0.3590181377659198

l2.norm(contrafacticos.iloc[0]-contrafacticos.iloc[1])
#contrafacticos sin cond lip: 0.4618856943250355
#contrafacticos sin cond lip: 0.0.35878291050228384
contrafacticos.to_csv('contrafacticos_conlip.csv',index=False)
x0.to_csv('x0inst6y7.csv',index=False)

