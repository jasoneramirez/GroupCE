# GroupCE

Code to implement the optimization models described in the Numerical Illustration section in paper *Mathematical Optimization Modelling for Group Counterfactual Explanations* by Emilio Carrizosa, Jasone Ramírez-Ayerbe and Dolores Romero Morales.
A preprint of the paper can be found here: 
### Requirements

To run the model, the gurobi solver is required. Free academics licenses are available. 


### Files

To run the experiments for the one-for-one model:

* 'model_traning.py': to train the classifier, either a random forest or a logistic regression model
* 'OforO_model_opt.py': define the optimization model (for the RF or LR) using pyomo
* 'OforO_run.py': to solve the optimization model
* 'OforO_visualization.py': to define x0, the cost function, the threshold nu for the probability, and to solve the problem and generate the heatmaps with the changes

To run the experiments for the one-for-many model: 

* 'OforM.py': calculates $P$ counterfactuals and their respective clusters for the LR model 

For for heatmaps: 'heatmaps.ipynb'
