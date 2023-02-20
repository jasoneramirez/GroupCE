# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:03:17 2020

@author: Jasone
"""

# el run

from __future__ import division
#import Pyomo
from pyomo.environ import *
from pyomo.opt import SolverFactory
from numpy import linalg as l2
# Cargar el modelo
import modelo_opt
import model_training
import numpy as np
import pandas as pd
import os
import math


def feature_type(dataset):
        features=list(dataset.columns)
        feat_type = ['Categorical' if x.name == 'category' else 'Numerical' for x in dataset.dtypes]
        features_type = dict(zip(features, feat_type))
        return features_type

def optimization_collective(x0,y0,perc,model,leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical,constraints_left_categorical,index_cont,index_cat,model_clas,tree_data,timelimit,x_ini,lam,nu):
           
    n_features=x0.shape[1]
    n_ind=x0.shape[0]
    n_trees=len(leaves)
    features=list(x0.columns)
    features_type=feature_type(x0)
    #indices=list(x0.index)

    nu2=n_trees*(2*nu-1)

    n_cat=sum(x0.dtypes=='category')
    n_cont=x0.shape[1]-n_cat

    x0_1={}
    x0_2={}

    cat_features=[f for f in features if features_type[f]=='Categorical']
    cont_features=[f for f in features if features_type[f]=='Numerical']

    for j in range(0,n_ind):
        for (f,i) in zip(cont_features,index_cont):
            x0_1[i,j]=x0.iloc[j][f]
    for j in range(0,n_ind):
        for (f,i) in zip(cat_features,index_cat):
            x0_2[i,j]=x0.iloc[j][f]

    y={}
    for j in range(0,n_ind):
        y[j]=-list(y0)[j]

    #initial solution:
    
    if isinstance(x_ini,pd.DataFrame):
        x_ini_1={}
        for j in range(0,n_ind):
            for (f,i) in zip(cont_features,index_cont):
                x_ini_1[i,j]=x_ini.iloc[j][f]

        x_ini_2={}   
        for j in range(0,n_ind):
            for (f,i) in zip(cat_features,index_cat):
                x_ini_2[i,j]=x_ini.iloc[j][f]

        sol_ini=model_clas.apply(x_ini)  #necesito el modelo
        z_sol_ini={}
        for i in range(0,n_ind):
            z_sol_ini[i]=list(map(lambda tl:tree_data[tl[0]]['index_leaves'].index(tl[1]),enumerate(sol_ini[i]))) #necesito datos_arbol


        z={}
        for i in range(0,n_ind):
            for t in range(n_trees):
                for l in range(leaves[t]):
                    z[t,l,i]=0
                    z[t,z_sol_ini[i][t],i]=1

        D={}
        for i in range(0,n_ind):
            for t in range(n_trees):
                D[t,i]=0
                for l in range(leaves[t]):
                    D[t,i]=D[t,i]+values[t][l]*z[t,l,i]



    leaf={}
    for t in range(n_trees):
        leaf[t]=leaves[t]


    n_left_num=len(constraints_left_numerical)
    n_right_num=len(constraints_right_numerical)
    n_left_cat=len(constraints_left_categorical)
    n_right_cat=len(constraints_right_categorical)

    restric_left_num={}
    for i in range(n_left_num):
        restric_left_num[i]=constraints_left_numerical[i]

    restric_right_num={}
    for i in range(n_right_num):
        restric_right_num[i]=constraints_right_numerical[i]

    restric_left_cat={}
    for i in range(n_left_cat):
        restric_left_cat[i]=constraints_left_categorical[i]

    restric_right_cat={}
    for i in range(n_right_cat):
        restric_right_cat[i]=constraints_right_categorical[i]


    values_leaf_dict={}
    for t in range(n_trees):
        for l in range(leaves[t]):
            values_leaf_dict[(t,l)]=values[t][l]

    data= {None: dict(
            #N0 = {None : 0},  #n variables innamovibles
            N1 = {None : n_cont}, #n variables continuas
            N2 = {None : n_cat},  #n variables categoricas
            #x0_0={None:0},
            ind ={None: n_ind},
            M1={None:1}, 
            M2={None:1}, 
            M3={None:1.1}, 
            #MAD={0: 0.3283222612880656,1: 1.0,2: 9.370046020955403,3: 0.12972769411924018,4: 0.5122390664936853,
            #        5: 28.984873371784516,6: 1.914261854423508, 7: 2.965204437011204, 8: 108.22996195090894,
            #        9: 1.7049925512814401,10: 8.095008113040556,11: 7.1090776377343605,12: 1.0},
            epsi={None:1e-6},
            trees = {None:n_trees},
            leaves = leaf,
            values_leaf=values_leaf_dict,
            nleft_num={None:n_left_num},
            nright_num={None:n_right_num},
            nleft_cat={None:n_left_cat},
            nright_cat={None:n_right_cat},
            left_num=restric_left_num,
            right_num=restric_right_num,
            left_cat=restric_left_cat,
            right_cat=restric_right_cat,
            x0_1=x0_1,
            x0_2=x0_2,
            y=y,
            perc={None: perc},
            lam={None:lam},
            nu={None:nu2}
            )}


  
    instance = model.create_instance(data) 
    opt= SolverFactory('gurobi', solver_io="python")
    opt.options['TimeLimit'] = timelimit

    if isinstance(x_ini,pd.DataFrame):

        for j in range(0,n_ind):
            for i in index_cont:
                instance.x_1[i,j]=x_ini_1[i,j]
    
        for j in range(0,n_ind):
            for i in index_cat:
                instance.x_2[i,j]=x_ini_2[i,j]

        for i in range(0,n_ind):
            for t in range(n_trees):
                for l in range(leaves[t]):
                    instance.z[t,l,i]=z[t,l,i]

        for i in range(0,n_ind):
            for t in range(n_trees):
                instance.D[t,i]=D[t,i]
      
    results = opt.solve(instance,tee=True) # tee=True: ver iteraciones por pantalla
    #results = opt.solve(instance,tee=True,warmstart=True)

    #comprobar que funciona

    x_sol_aux=np.zeros([len(features),n_ind])

    for i in index_cont:
        for j in instance.I:
            x_sol_aux[i,j]=instance.x_1[i,j].value
    for i in index_cat:
        for j in instance.I:
            x_sol_aux[i,j]=instance.x_2[i,j].value
    
    #for (i,k) in zip(instance.Cont,index_cont):
    #    for j in instance.I:
    #        x_sol_aux[k,j]=instance.x_1[i,j].value
    #for (i,k) in zip(instance.Cat,index_cat):
    #    for j in instance.I:
    #        x_sol_aux[k,j]=instance.x_2[i,j].value

   
    #xi2={}
    #for i in instance.Cont:
    #    xi2[i]=instance.xi2[i].value

    #xi3={}
    #for i in instance.Cat:
    #    xi3[i]=instance.xi3[i].value

    x_sol=pd.DataFrame(x_sol_aux,features)

    #print(x_sol)

    #print(xi2)
    #print(xi3)

    cambio_x=np.zeros([len(features),n_ind])
    

    for i in range(len(features)):
        for j in range(n_ind):
            cambio_x[i,j]=x_sol_aux[i,j]-x0.iloc[j,i]
            if cambio_x[i,j]<=1e-10:
                cambio_x[i,j]==0
    

    data=pd.DataFrame(cambio_x,index=features)

    data=data.transpose()

    #objective_value=instance.obj.expr()

    #if objective=="l2l0ind":
    #   model.lam*(sum(model.xi[n,i] for n in model.Cont for i in model.I)+sum((model.x0_2[n,i]-model.x_2[n,i])**2 for n in model.Cat for i in model.I))+(sum( (model.x0_1[n,i]-model.x_1[n,i])**2 for n in model.Cont for i in model.I))
        


    #l2 + l0 global
    #elif objective=="l2l0global":
       
    objective_value= instance.lam.value*(sum(instance.xi2[n].value for n in list(instance.Cont.data()))+sum(instance.xi3[n].value for n in list(instance.Cat.data())))+sum( (instance.x0_1[n,i]-instance.x_1[n,i].value)**2 for n in list(instance.Cont.data()) for i in list(instance.I.data()))
        

    #with open('sol_crim100_5de10_l0_global.pkl', 'wb') as f:
    #    pickle.dump(data, f)
    
    

    return data, objective_value


def optimization_individual(i,timelimit,x0,y0,lam,model,leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical,constraints_left_categorical,index_cont,index_cat,model_clas,tree_data,x,y):
     
    n_features=x0.shape[1]
    n_ind=x0.shape[0]
    n_trees=len(leaves)
    features=list(x0.columns)
    features_type=feature_type(x0)
    n_cat=sum(x0.dtypes=='category')
    n_cont=x0.shape[1]-n_cat


    x0=x0.loc[i]
    y0=y0[i]
    print(y0)

    x0_1={}
    x0_2={}

    cat_features=[f for f in features if features_type[f]=='Categorical']
    cont_features=[f for f in features if features_type[f]=='Numerical']

    for (f,i) in zip(cont_features,index_cont):
         x0_1[i]=x0[f]
    for (f,i) in zip(cat_features,index_cat):
         x0_2[i]=x0[f]


    
    distance=100
    for i in range(x.shape[0]):
        if y.iloc[i]==-y0: #y_pred
            d=l2.norm(x0-x.iloc[i])
            if d<=distance:
                distance=d
                x_ini=x.iloc[i] #initial solution
    

    x_ini_1={}
    x_ini_2={}
    for (f,i) in zip(cont_features,index_cont):
        x_ini_1[i]=x_ini[f]
    for (f,i) in zip(cat_features,index_cat):
         x_ini_2[i]=x_ini[f]
    

    x_ini=np.array(x_ini).reshape(1,-1)
    sol_ini=model_clas.apply(x_ini)[0]  
    z_sol_ini=list(map(lambda tl:tree_data[tl[0]]['index_leaves'].index(tl[1]),enumerate(sol_ini))) #necesito datos_arbol

    z={}
    for t in range(n_trees):
        for l in range(leaves[t]):
            z[t,l]=0
            z[t,z_sol_ini[t]]=1

    D={}
    for t in range(n_trees):
        D[t]=0
        for l in range(leaves[t]):
            D[t]=D[t]+values[t][l]*z[t,l]

  

    leaf={}
    for t in range(n_trees):
        leaf[t]=leaves[t]


    n_left_num=len(constraints_left_numerical)
    n_right_num=len(constraints_right_numerical)
    n_left_cat=len(constraints_left_categorical)
    n_right_cat=len(constraints_right_categorical)

    restric_left_num={}
    for i in range(n_left_num):
        restric_left_num[i]=constraints_left_numerical[i]

    restric_right_num={}
    for i in range(n_right_num):
        restric_right_num[i]=constraints_right_numerical[i]

    restric_left_cat={}
    for i in range(n_left_cat):
        restric_left_cat[i]=constraints_left_categorical[i]

    restric_right_cat={}
    for i in range(n_right_cat):
        restric_right_cat[i]=constraints_right_categorical[i]


    values_leaf_dict={}
    for t in range(n_trees):
        for l in range(leaves[t]):
            values_leaf_dict[(t,l)]=values[t][l]

    data= {None: dict(
            #N0 = {None : 0},  #n variables innamovibles
            N1 = {None : n_cont}, #n variables continuas
            N2 = {None : n_cat},  #n variables categoricas
            #x0_0={None:0},
            M1={None:1.1}, 
            M2={None:1.1}, 
            M3={None:1.1}, 
            #MAD={0: 0.3283222612880656,1: 1.0,2: 9.370046020955403,3: 0.12972769411924018,4: 0.5122390664936853,
            #        5: 28.984873371784516,6: 1.914261854423508, 7: 2.965204437011204, 8: 108.22996195090894,
            #        9: 1.7049925512814401,10: 8.095008113040556,11: 7.1090776377343605,12: 1.0},
            epsi={None:1e-6},
            trees = {None:n_trees},
            leaves = leaf,
            values_leaf=values_leaf_dict,
            nleft_num={None:n_left_num},
            nright_num={None:n_right_num},
            nleft_cat={None:n_left_cat},
            nright_cat={None:n_right_cat},
            left_num=restric_left_num,
            right_num=restric_right_num,
            left_cat=restric_left_cat,
            right_cat=restric_right_cat,
            x0_1=x0_1,
            x0_2=x0_2,
            y={None:-y0},
            lam={None: lam}
            )}


  
    instance = model.create_instance(data) 
    opt= SolverFactory('gurobi', solver_io="python")
    opt.options['TimeLimit'] = timelimit

    for i in index_cont:
        instance.x_1[i]=x_ini_1[i]

    for i in index_cat:
        instance.x_2[i]=x_ini_2[i]

    for t in range(n_trees):
        for l in range(leaves[t]):
            instance.z[t,l]=z[t,l]

    for t in range(n_trees):
        instance.D[t]=D[t]

      
    results = opt.solve(instance,tee=True,warmstart=True) # tee=True: ver iteraciones por pantalla
   
 
    x_sol_aux=np.zeros(len(features))
    for i in index_cont:
        x_sol_aux[i]=instance.x_1[i].value
    for i in index_cat:
        x_sol_aux[i]=instance.x_2[i].value
   



    x_sol=pd.DataFrame(x_sol_aux,features,columns=['x'])

    #print(x_sol)

    salida=-y0*sum([instance.D[t].value for t in range(n_trees)])
    print('salida: '+str(salida))
    #print(instance)

    #print(xi2)
    #print(xi3)

    cambio_x=np.zeros(len(features))

    for j in range(len(features)):
        cambio_x[j]=x_sol_aux[j]-x0[j]
    


 

    #with open('sol_crim100_5de10_l0_global.pkl', 'wb') as f:
    #    pickle.dump(data, f)
    
    

    return x_sol, cambio_x, -y0



def optimization_lineal_collective(x0,y0,perc,model,w,b,index_cont,index_cat,timelimit,lam,nu):
           
    n_features=x0.shape[1]
    n_ind=x0.shape[0]
    features=list(x0.columns)
    features_type=feature_type(x0)
    #indices=list(x0.index)

    nu2=-math.log(1/nu-1)

    n_cat=sum(x0.dtypes=='category')
    n_cont=x0.shape[1]-n_cat

    x0_1={}
    x0_2={}

    cat_features=[f for f in features if features_type[f]=='Categorical']
    cont_features=[f for f in features if features_type[f]=='Numerical']

    for j in range(0,n_ind):
        for (f,i) in zip(cont_features,index_cont):
            x0_1[i,j]=x0.iloc[j][f]
    for j in range(0,n_ind):
        for (f,i) in zip(cat_features,index_cat):
            x0_2[i,j]=x0.iloc[j][f]

    y={}
    for j in range(0,n_ind):
        y[j]=-list(y0)[j]

    k=0; # y(wx+b)>=k

    w_dict={}
    for i in range(len(features)):
        w_dict[i]=w[0][i]

    bound=abs(sum(w[0])+b)[0]

    data= {None: dict(
            #N0 = {None : 0},  #n variables innamovibles
            N1 = {None : n_cont}, #n variables continuas
            N2 = {None : n_cat},  #n variables categoricas
            #x0_0={None:0},
            ind ={None: n_ind},
            M3={None:1.1}, 
            k = {None: nu2},
            w=w_dict,
            b={None: b[0]},
            x0_1=x0_1,
            x0_2=x0_2,
            y=y,
            perc={None: perc},
            bound={None: bound},
            lam={None:lam}
            )}


  
    instance = model.create_instance(data) 
    opt= SolverFactory('gurobi', solver_io="python")
    opt.options['TimeLimit'] = timelimit

      
    results = opt.solve(instance,tee=True) # tee=True: ver iteraciones por pantalla
    #results = opt.solve(instance,tee=True,warmstart=True)

    #comprobar que funciona

    x_sol_aux=np.zeros([len(features),n_ind])

    for i in index_cont:
        for j in instance.I:
            x_sol_aux[i,j]=instance.x_1[i,j].value
    for i in index_cat:
        for j in instance.I:
            x_sol_aux[i,j]=instance.x_2[i,j].value
    
    #for (i,k) in zip(instance.Cont,index_cont):
    #    for j in instance.I:
    #        x_sol_aux[k,j]=instance.x_1[i,j].value
    #for (i,k) in zip(instance.Cat,index_cat):
    #    for j in instance.I:
    #        x_sol_aux[k,j]=instance.x_2[i,j].value

   
    #xi2={}
    #for i in instance.Cont:
    #    xi2[i]=instance.xi2[i].value

    #xi3={}
    #for i in instance.Cat:
    #    xi3[i]=instance.xi3[i].value

    x_sol=pd.DataFrame(x_sol_aux,features)

    print(x_sol)

    #print(xi2)
    #print(xi3)

    cambio_x=np.zeros([len(features),n_ind])
    

    for i in range(len(features)):
        for j in range(n_ind):
            cambio_x[i,j]=x_sol_aux[i,j]-x0.iloc[j,i]
            if cambio_x[i,j]<=1e-10:
                cambio_x[i,j]==0
    

    data=pd.DataFrame(cambio_x,index=features)

    data=data.transpose()

    objective_value= instance.lam.value*(sum(instance.xi2[n].value for n in list(instance.Cont.data()))+sum(instance.xi3[n].value for n in list(instance.Cat.data())))+sum( (instance.x0_1[n,i]-instance.x_1[n,i].value)**2 for n in list(instance.Cont.data()) for i in list(instance.I.data()))
        

    #with open('sol_crim100_5de10_l0_global.pkl', 'wb') as f:
    #    pickle.dump(data, f)
    
    

    return data, objective_value



def optimization_lineal_ind(i,x0,y0,lam,model,w,b,index_cont,index_cat):
    # Ruta carpeta donde estás trabajando
    #path = "C:/Users/jas_r/OneDrive/Documentos/PhD/codigo"  
    #os.chdir(path)
    features=list(x0.columns)
    features_type=feature_type(x0)
    cat_features=[f for f in features if features_type[f]=='Categorical']
    cont_features=[f for f in features if features_type[f]=='Numerical']
    n_cont=len(cont_features)
    n_cat=len(cat_features)

    x0=x0.loc[i]
    y0=y0[i]

   

    x0_1={}
    x0_2={}

    for (f,i) in zip(cont_features,index_cont):
         x0_1[i]=x0[f]
    for (f,i) in zip(cat_features,index_cat):
         x0_2[i]=x0[f]


    k=0; # y(wx+b)>=k

    w_dict={}
    for i in range(len(features)):
        w_dict[i]=w[0][i]

    data= {None: dict(
            #N0 = {None : 0},  #n variables innamovibles
            N1 = {None : n_cont}, #n variables continuas
            N2 = {None : n_cat},  #n variables categoricas
            #x0_0={None:0},
            x0_1=x0_1,
            x0_2=x0_2,
            M3={None:2},
            y={None: -y0},
            k = {None: 0},
            w=w_dict,
            b={None: b[0]},
            lam={None: lam}
            )}


    #print(data)
 
    instance = model.create_instance(data) 
    opt= SolverFactory('gurobi', solver_io="python")       
    results = opt.solve(instance,tee=True) 

    x_sol_aux=np.zeros(len(features))
    for i in index_cont:
        x_sol_aux[i]=instance.x_1[i].value
    for i in index_cat:
        x_sol_aux[i]=instance.x_2[i].value
   


    x_sol=pd.DataFrame(x_sol_aux,features,columns=['x'])

    #with open("resultados_SVM_l0l2.txt", "a") as f:
            #print('x0:\n'+str(x0),file=f)
            #print('y0: '+str(y0),file=f)
            #print('x:\n'+str(x_sol),file=f)
            #print('y: '+str(-y0),file=f)
            #print('Distancia: '+str(instance.obj()),file=f)
            #print('Tiempo '+str(results['Solver']._list[0]['Wallclock time']),file=f)
            #print(str(individuo)+','+str(instance.obj()),file=f)
    cambio_x=np.zeros(len(features))

    for j in range(len(features)):
        cambio_x[j]=x_sol_aux[j]-x0[j]

    #print(cambio_x)

    return x_sol, cambio_x, -y0