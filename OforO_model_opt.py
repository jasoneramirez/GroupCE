# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:03:30 2020

@author: Jasone
"""



from __future__ import division
from pyomo.environ import *



def modelo_opt_rf_nonseparable(leaves,index_cont,index_cat,objective,cond_dist):

    n_trees=len(leaves)
    model = AbstractModel()

    index_list = [] #index list (tree,leaf)
    for t in range(n_trees):
       for l in range(leaves[t]):
           index_list.append((t,l))


    #parameters
    model.N1 = Param( within=PositiveIntegers ) #continuos variables
    model.N2 = Param( within=PositiveIntegers ) #categorical variables
    model.ind =Param (within=PositiveIntegers) #number of individuals
    model.Cont=Set(dimen=1,initialize=index_cont)
    model.Cat=Set(dimen=1,initialize=index_cat)
    model.I=RangeSet(0,model.ind-1)

    model.trees= Param (within =PositiveIntegers) #number of trees
    model.t=RangeSet(0,model.trees-1)
    model.leaves = Param(model.t) #number of leaves of each tree
    model.index_list = Set(dimen=2,initialize=index_list)
    model.values_leaf=Param(model.index_list)

    #parameters constraints
    model.nleft_num=Param(within=PositiveIntegers) 
    model.nright_num=Param (within=PositiveIntegers) 
    model.nleft_cat=Param(within=PositiveIntegers)
    model.nright_cat=Param(within=PositiveIntegers)
    model.resleft_num=RangeSet(0,model.nleft_num-1) 
    model.resright_num= RangeSet(0,model.nright_num-1)
    model.resleft_cat=RangeSet(0,model.nleft_cat-1) 
    model.resright_cat= RangeSet(0,model.nright_cat-1)
    model.left_num=Param(model.resleft_num,within=Any)
    model.right_num=Param(model.resright_num,within=Any)
    model.left_cat=Param(model.resleft_cat,within=Any)
    model.right_cat=Param(model.resright_cat,within=Any)


    #bigMs
    model.M1=Param(within=PositiveReals) 
    model.M2=Param(within=Reals) 
    model.M3=Param(within=PositiveReals)

    model.epsi=Param(within=PositiveReals) #epsilon
    model.nu=Param(within=Reals)


    model.y= Param( model.I,within=Integers) # y=-y0 each ind
    model.x0_1=Param(model.Cont,model.I) #continuos x0
    model.x0_2=Param(model.Cat,model.I) #categorical x0
    model.perc= Param(within=Integers) #number of individuals to be changed
    model.lam=Param(within=Reals)


    #variables
    model.x_1 = Var( model.Cont, model.I, bounds=(0,1) ) 
    model.x_2 =Var (model.Cat,model.I, within=Binary) 


    model.z=Var(model.index_list,model.I,within=Binary)
    model.D=Var(model.t,model.I) 
    model.xi=Var(model.Cont,model.I,within=Binary)
    model.xi2=Var(model.Cont, within=Binary) #l0 global
    model.xi3=Var(model.Cat, within=Binary) 
    model.w=Var(model.I,within=Binary) #w=1 if ind changes class
    model.phi=Var(model.I,within=Reals) #linearization auxiliar




    if objective=="l2l0ind":
        def obj_rule(model):
            return model.lam*(sum(model.xi[n,i] for n in model.Cont for i in model.I)+sum((model.x0_2[n,i]-model.x_2[n,i])**2 for n in model.Cat for i in model.I))+(sum( (model.x0_1[n,i]-model.x_1[n,i])**2 for n in model.Cont for i in model.I))
        model.obj = Objective( rule=obj_rule )


    #l2 + l0 global
    elif objective=="l2l0global":
        def obj_rule(model):
            return model.lam*(sum(model.xi2[n] for n in model.Cont)+sum(model.xi3[n] for n in model.Cat))+sum( (model.x0_1[n,i]-model.x_1[n,i])**2 for n in model.Cont for i in model.I)
        model.obj = Objective( rule=obj_rule )



    #constraints


    def path_left_num(model,s,i):
        return model.x_1[model.left_num[s][2],i]-(model.M1-model.left_num[s][3])*(1-model.z[model.left_num[s][0],model.left_num[s][1],i])+model.epsi<=model.left_num[s][3]
    model.pathleft_num= Constraint(model.resleft_num,model.I,rule=path_left_num)

    def path_right_num(model,s,i):
        return model.x_1[model.right_num[s][2],i]+(model.M2+model.right_num[s][3])*(1-model.z[model.right_num[s][0],model.right_num[s][1],i])-model.epsi>=model.right_num[s][3]
    model.pathright_num= Constraint(model.resright_num,model.I,rule=path_right_num)


    def path_left_cat(model,s,i):
        return model.x_2[model.left_cat[s][2],i]-(model.M1-model.left_cat[s][3])*(1-model.z[model.left_cat[s][0],model.left_cat[s][1],i])+model.epsi<=model.left_cat[s][3]
    model.pathleft_cat= Constraint(model.resleft_cat,model.I,rule=path_left_cat)

    def path_right_cat(model,s,i):
        return model.x_2[model.right_cat[s][2],i]+(model.M2+model.right_cat[s][3])*(1-model.z[model.right_cat[s][0],model.right_cat[s][1],i])-model.epsi>=model.right_cat[s][3]
    model.pathright_cat= Constraint(model.resright_cat,model.I,rule=path_right_cat)


    def one_path(model,t,i):
        return sum(model.z[t,l,i] for l in RangeSet(0,model.leaves[t]-1))==1.0
    model.path=Constraint(model.t,model.I,rule=one_path)

    def def_output(model,t,i):
        return model.D[t,i]==sum(model.values_leaf[t,l]*model.z[t,l,i] for l in RangeSet(0,model.leaves[t]-1))
    model.output=Constraint(model.t,model.I,rule=def_output)

    def def_finalclass(model,i):
        return model.y[i]*model.phi[i]>=model.nu
    model.finalclass=Constraint(model.I, rule=def_finalclass) 

    def aux_group1(model,i):
        return -model.w[i]*model.trees<=model.phi[i]
    model.auxg1=Constraint(model.I, rule=aux_group1)

    def aux_group2(model,i):
        return model.phi[i]<=model.w[i]*model.trees
    model.auxg2=Constraint(model.I, rule=aux_group2)

    def aux_group3(model,i):
        return sum(model.D[t,i] for t in model.t)-(1-model.w[i])*model.trees<=model.phi[i]
    model.auxg3=Constraint(model.I, rule=aux_group3)

    def aux_group4(model,i):
        return model.phi[i]<=sum(model.D[t,i] for t in model.t)+(1-model.w[i])*model.trees
    model.auxg4=Constraint(model.I, rule=aux_group4)

    def ind_change(model):
        return sum(model.w[i] for i in model.I)>=model.perc #number of individuals to be changed
    model.indcambio=Constraint(rule=ind_change)

    def aux_l01(model,n,i):
        return -model.M3*model.xi[n,i]<=(model.x_1[n,i]-model.x0_1[n,i])
    model.auxl01=Constraint(model.Cont,model.I,rule=aux_l01)

    def aux_l02(model,n,i):
        return (model.x_1[n,i]-model.x0_1[n,i])<=model.xi[n,i]*model.M3
    model.auxl02=Constraint(model.Cont,model.I,rule=aux_l02)


    def aux_l0global1(model,n,i):
        return model.xi2[n] >= model.xi[n,i]
    model.auxl0g1=Constraint(model.Cont,model.I,rule=aux_l0global1)

    def aux_l0global2(model,n,i):
        return model.xi3[n] >= (model.x0_2[n,i]-model.x_2[n,i])**2
    model.auxl0g2=Constraint(model.Cat,model.I,rule=aux_l0global2)


    if cond_dist=='True':
        def condlip(model,i,j):
            return sum((model.x_2[n,i]-model.x_2[n,j])**2 for n in model.Cat)+sum((model.x_1[n,i]-model.x_1[n,j])**2 for n in model.Cont)<=10*sum((model.x0_2[n,i]-model.x0_2[n,j])**2 for n in model.Cat)+sum((model.x0_1[n,i]-model.x0_1[n,j])**2 for n in model.Cont)
        model.condlipdist=Constraint(model.I,model.I,rule=condlip)

    return model





def modelo_opt_lineal_nonseparable(index_cont,index_cat,objective):

    
    model = AbstractModel()


    #parameters
    model.N1 = Param( within=PositiveIntegers ) #continuos variables
    model.N2 = Param( within=PositiveIntegers ) #categorical variables
    model.ind =Param (within=PositiveIntegers) #number of individuals
    model.Cont=Set(dimen=1,initialize=index_cont)
    model.Cat=Set(dimen=1,initialize=index_cat)
    model.I=RangeSet(0,model.ind-1)

    #model parameters
    model.w = Param( RangeSet(0,model.N1+model.N2) ) #weights
    model.b = Param( within=Reals ) #bias
    model.k = Param( within=Reals) #threshold

    


    #bigMs
    model.M3=Param(within=PositiveReals)
    model.bound=Param(within=Reals)



    model.y= Param( model.I,within=Integers) # y=-y0 each ind
    #model.x0_0 =Param(model.Fij) #unmovable x0
    model.x0_1=Param(model.Cont,model.I) #continuos x0
    model.x0_2=Param(model.Cat,model.I) #categorical x0
    model.perc= Param(within=Integers) #number of individuals to be changed
    model.lam=Param(within=Reals)


    #variables
    model.x_1 = Var( model.Cont, model.I, bounds=(0,1) ) 
    model.x_2 =Var (model.Cat,model.I, within=Binary) 


   
    model.xi=Var(model.Cont,model.I,within=Binary)
    model.xi2=Var(model.Cont, within=Binary) #l0 global
    model.xi3=Var(model.Cat, within=Binary) 
    model.q=Var(model.I,within=Binary) #q=1 if ind changes class
    model.phi=Var(model.I,within=Reals) #linearization auxiliar



    if objective=="l2l0ind":
        def obj_rule(model):
            return model.lam*(sum(model.xi[n,i] for n in model.Cont for i in model.I)+sum((model.x0_2[n,i]-model.x_2[n,i])**2 for n in model.Cat for i in model.I))+(sum( (model.x0_1[n,i]-model.x_1[n,i])**2 for n in model.Cont for i in model.I))
        model.obj = Objective( rule=obj_rule )

    elif objective=="l2l0global":
    #l2 + l0 global
        def obj_rule(model):
            return model.lam*(sum(model.xi2[n] for n in model.Cont)+sum(model.xi3[n] for n in model.Cat))+sum( (model.x0_1[n,i]-model.x_1[n,i])**2 for n in model.Cont for i in model.I)
        model.obj = Objective( rule=obj_rule )


    #constraints


    
    def clase_rule(model,i):
        return  model.y[i]*model.phi[i]>=model.k
    model.clase = Constraint (model.I,rule=clase_rule)

  


    def aux_group1(model,i):
        return -model.q[i]*model.bound<=model.phi[i]
    model.auxg1=Constraint(model.I, rule=aux_group1)

    def aux_group2(model,i):
        return model.phi[i]<=model.q[i]*model.bound
    model.auxg2=Constraint(model.I, rule=aux_group2)

    def aux_group3(model,i):
        return (sum(model.w[n]*model.x_1[n,i] for n in model.Cont)+sum(model.w[s]*model.x_2[s,i] for s in model.Cat)+model.b)-(1-model.q[i])*model.bound<=model.phi[i]
    model.auxg3=Constraint(model.I, rule=aux_group3)

    def aux_group4(model,i):
        return model.phi[i]<=(sum(model.w[n]*model.x_1[n,i] for n in model.Cont)+sum(model.w[s]*model.x_2[s,i] for s in model.Cat)+model.b)+(1-model.q[i])*model.bound
    model.auxg4=Constraint(model.I, rule=aux_group4)

    def ind_change(model):
        return sum(model.q[i] for i in model.I)>=model.perc #number of individuals to be changed
    model.indcambio=Constraint(rule=ind_change)

    def aux_l01(model,n,i):
        return -model.M3*model.xi[n,i]<=(model.x_1[n,i]-model.x0_1[n,i])
    model.auxl01=Constraint(model.Cont,model.I,rule=aux_l01)

    def aux_l02(model,n,i):
        return (model.x_1[n,i]-model.x0_1[n,i])<=model.xi[n,i]*model.M3
    model.auxl02=Constraint(model.Cont,model.I,rule=aux_l02)


    def aux_l0global1(model,n,i):
        return model.xi2[n] >= model.xi[n,i]
    model.auxl0g1=Constraint(model.Cont,model.I,rule=aux_l0global1)

    def aux_l0global2(model,n,i):
        return model.xi3[n] >= (model.x0_2[n,i]-model.x_2[n,i])**2
    model.auxl0g2=Constraint(model.Cat,model.I,rule=aux_l0global2)





    return model


