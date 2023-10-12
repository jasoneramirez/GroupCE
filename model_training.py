import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import random


#logistic regression or svm with linear kernel



def linear_model(type,x,y):

    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.33,random_state=0)

    if type=='LR':
        model = LogisticRegression(solver='liblinear', random_state=0,C=10.0) #choose C by cv
        model.fit(x_train,y_train)
        y_pred_prob=model.predict_proba(x_test)
        y_pred=pd.DataFrame(model.predict(x).tolist(),index=x.index,columns=['y'])
        w=model.coef_
        b=model.intercept_
        
    elif tipo=='SVM':
        model=SVC()
        parameters = {'kernel':['linear'],'gamma':np.logspace(-1, 1, 5)} #add more if wanted
        grid_sv = GridSearchCV(model, param_grid=parameters)
        grid_sv.fit(x_train, y_train)
        model=grid_sv.best_estimator_
        #print("Best classifier :", grid_sv.best_estimator_)
        #model=SVC(kernel="linear",gamma=0.01) #if wanted
        #model.fit(x_train,y_train)
        y_pred=pd.DataFrame(model.predict(x).tolist(),index=x.index,columns=['y'])
        w=model.coef_
        b=model.intercept_

        
    return x,y,y_pred,model,w, b
    

def randomforest(n_trees,maxdepth,x,y):

    cols = x.columns
    num_cols = x._get_numeric_data().columns
    cat_cols=list(set(cols) - set(num_cols))
    categ=[i for i, j in enumerate(cols) if j in cat_cols] #index categorical variables

    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.33,random_state=0)
    index=x_train.index #to know x_train and x_test 



    model=RandomForestClassifier(n_estimators=n_trees,random_state=0,max_depth=maxdepth)
    model.fit(x_train, y_train)

    trees=[] 
    for t in range(n_trees):
        trees.append(model.estimators_[t]) 

    tree_parameters={}
    for t in range(len(trees)):
        tree_parameters[t]={}
        tree_parameters[t]['n_nodes']=trees[t].tree_.node_count
        tree_parameters[t]['children_left'] = trees[t].tree_.children_left
        tree_parameters[t]['children_right'] = trees[t].tree_.children_right
        tree_parameters[t]['feature'] = trees[t].tree_.feature
        tree_parameters[t]['threshold'] = trees[t].tree_.threshold

    def child_node(x,t):
        if x in tree_data[t]['parents_l'].keys():
            child_node=tree_data[t]['parents_l'][x]
            where='l'
        else:
            child_node=tree_data[t]['parents_r'][x]
            where='r'
        return([child_node,where])

    def aux_list(x,t):
        k=x
        dict={'left':[],'right':[]}
        while k!=0:
            node=child_node(k,t)
            if node[1]=='l':
                dict['left'].append(node[0])
                k=node[0]
            else:
                dict['right'].append(node[0])
                k=node[0]
        return dict   

    def threshold_feature(leaf,direccion,t):
        d=tree_data[t]['paths'][leaf]
        res=[]
        if direccion=='left':
            s=d['left']
            for i in s:
                res.append([tree_parameters[t]['feature'][i],tree_parameters[t]['threshold'][i]])     
        else:
            s=d['right']
            for i in s:
                res.append([tree_parameters[t]['feature'][i],tree_parameters[t]['threshold'][i]])    
        return res
    
   
  
     

    tree_data={}
    classes=np.array([-1,1])
    for t in range(len(trees)):
        tree_data[t]={}
        is_leaves = np.zeros(shape=tree_parameters[t]['n_nodes'], dtype=bool)
        is_leaves=tree_parameters[t]['children_left']==tree_parameters[t]['children_right']
        tree_data[t]['index_leaves']=np.where(is_leaves==True)[0].tolist()
        tree_data[t]['value_leaves']=[classes[np.argmax(i)] for i in trees[t].tree_.value[tree_data[t]['index_leaves']]]
        tree_data[t]['index_splits']=np.where(is_leaves==False)[0].tolist()
        tree_data[t]['n_leaves']=len(tree_data[t]['index_leaves'])
        parents_l={}
        parents_r={}
        for j in range(tree_parameters[t]['n_nodes']):
            if len(np.where(tree_parameters[t]['children_left']==j)[0])!=0:
                parents_l[j]=np.where(tree_parameters[t]['children_left']==j)[0][0]
        for j in range(tree_parameters[t]['n_nodes']):
            if len(np.where(tree_parameters[t]['children_right']==j)[0])!=0:
                parents_r[j]=np.where(tree_parameters[t]['children_right']==j)[0][0]
        tree_data[t]['parents_l']=parents_l
        tree_data[t]['parents_r']=parents_r
        tree_data[t]['paths']={}
        for k in tree_data[t]['index_leaves']:
            tree_data[t]['paths'][k]=aux_list(k,t)

    constraints={}
    for t in range(len(trees)):
        constraints[t]={'left':[],'right':[]}
        for i in tree_data[t]['index_leaves']:
            constraints[t]['left'].append(threshold_feature(i,'left',t))   
            constraints[t]['right'].append(threshold_feature(i,'right',t))

    constraints_left=[]
    for i in range(n_trees):
        l=constraints[i]['left']
        for j in range(tree_data[i]['n_leaves']):
            l2=l[j]
            for k in range(len(l2)):
                l3=l2[k]
                constraints_left.append((i,j,l3[0],l3[1]))
            
    constraints_right=[]
    for i in range(n_trees):
        l=constraints[i]['right']
        for j in range(tree_data[i]['n_leaves']):
            l2=l[j]
            for k in range(len(l2)):
                l3=l2[k]
                constraints_right.append((i,j,l3[0],l3[1]))
    
    constraints_right_categorical=[]
    constraints_left_categorical=[]
    for res in constraints_right:
        if res[2] in categ:
            constraints_right_categorical.append(res)
    for res in constraints_left:
        if res[2] in categ:
            constraints_left_categorical.append(res)
    constraints_right_numerical=[x for x in constraints_right if x not in constraints_right_categorical]
    constraints_left_numerical=[x for x in constraints_left if x not in constraints_left_categorical]

    values={}
    for i in range(n_trees):
        values[i]=tree_data[i]['value_leaves']

    leaves=[]
    for i in range(n_trees):
        leaves.append(tree_data[i]['n_leaves'])

    y_pred=pd.DataFrame(model.predict(x).tolist(),index=x.index,columns=['y'])
    y_pred_prob=model.predict_proba(x)

   
   
   
    return leaves, values, constraints_right_numerical, constraints_left_numerical, constraints_right_categorical,constraints_left_categorical, index, x, y, y_pred, y_pred_prob, model, tree_data
