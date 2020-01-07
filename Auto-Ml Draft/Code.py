from tkinter import *
import tkinter
master = Tk() 
master.title("AUTOMATED MACHINE LEARNING")
#creating a text label
Label(master, text="AUTOMATED MACHINE LEARNING SYSTEM",font=("times new roman",20),fg="white",bg="maroon",height=2).grid(row=0,rowspan=2,columnspan=8,sticky=N+E+W+S,padx=5,pady=5)
location=StringVar()

Label(master, text='ENTER THE LOCATION OF DATASET').grid(row=2) 
e1 = Entry(master,textvariable=location) 
e1.grid(row=2,column=1) 
y=StringVar()
Label(master, text='ENTER THE PREDICTION COLUMN').grid(row=3) 
e1 = Entry(master,textvariable=y) 
e1.grid(row=3,column=1) 
reg = IntVar() 
Checkbutton(master, text='REGRESSION', variable=reg).grid(row=4, sticky=W) 
clas = IntVar() 
Checkbutton(master, text='CLASSIFICATION', variable=clas).grid(row=5, sticky=W) 
Label(master, text='TRAIN_TEST_SPLIT_SIZE [0-100]').grid(row=4,column=1) 
Label(master, text='NUMBER OF FOLDS').grid(row=5,column=1)
train_test=IntVar()
num_folds=IntVar()
e1 = Entry(master,textvariable=train_test) 
e2 = Entry(master,textvariable=num_folds) 
e1.grid(row=4, column=2) 
e2.grid(row=5, column=2)
Label(master, text='SCORING METRICS').grid(row=6) 
mse = IntVar() 
Checkbutton(master, text='MSE', variable=mse).grid(row=7, sticky=W) 
mae = IntVar() 
Checkbutton(master, text='MAE', variable=mae).grid(row=8, sticky=W) 
acc = IntVar() 
Checkbutton(master, text='ACCURACY', variable=acc).grid(row=9, sticky=W)
mainloop()
print(reg.get(),clas.get(),num_folds.get(),train_test.get(),mse.get(),mae.get(),acc.get(),y.get(),location.get())
machinelearning(location.get(),y.get(),train_test.get(),num_folds.get(),reg.get(),'mae')
def machinelearning(X,Y,train_test,num_folds,reg,metrics):
    def importregressionmodels():
        from sklearn.linear_model import LinearRegression
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor

        models = []
        models.append(('LR', LinearRegression()))
        models.append(('KNN', KNeighborsRegressor()))
        models.append(('CART', DecisionTreeRegressor()))
        models.append(('RF', RandomForestRegressor()))
        return models
    def importclassificationmodels():
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('RF', RandomForestClassifier()))
        models.append(('NB',GaussianNB()))
        return models
 
## IMPORT PANDAS AND NUMPY  
    import pandas as pd
    import numpy as np
    ## SPLIT THE DATA INTO TRAIN AND TEST
    data=pd.read_csv(X)
    x=data.drop(Y,axis=1)
    y=data[[Y]]
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=train_test)
    print('Runnng BASELINE models')
    d=reg
    if d==0:
 ## INTIALISING MODELS
        models=importclassificationmodels()

        results = []
        names = []
        from sklearn.model_selection import KFold
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score

## FOR BINARY AND MULTICLASS CLASSIFICATION:
        #print('please enter a number 1 -> Binary Classification    2 -> Multiclass Classification') 
        #temp=int(input())
        if reg==0:
## FOR BINARY CLASSIFICATION
 
## INPUT FOR KFOLD CROSS VALIDATION AND SCORING METRICS
            print('ENTER num_folds FOR KFOLD CROSS VALIDATION')
            num_folds = int(input())
            seed = 77
            print('ENTER THE SCORING METRICS 1 -> PRECISION 2 -> RECALL 3 -> F1-SCORE 4 -> ACCURACY')
            f=int(input())
            if f==1:
                scoring='precision'
            elif f==2:
                scoring='recall'
            elif f==3:
                scoring='f1'
            elif f==4:
                scoring='accuracy'
        elif temp==2:
            print('ENTER num_folds FOR KFOLD CROSS VALIDATION')
            num_folds = int(input())
            seed = 77
            print('ENTER THE SCORING METRICS 1 -> precision_macro 2 -> precision_micro 3 -> F1-recall_macro 4 -> accuracy')
            f=int(input())
            if f==1:
                scoring='precision_macro'
            elif f==2:
                scoring='precision_micro'
            elif f==3:
                scoring='recall_macro'
            elif f==4:
                scoring='recall_micro'
            elif f==5:
                scoring='accuracy'

        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import accuracy_score
        for name, model in models:
            kfold = KFold(n_splits=num_folds, random_state=seed)
            cv_results = cross_val_score(model, x_train, y_train,scoring=scoring, cv=kfold)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

### PARAMETER TUNING GRID SEARCH############
 
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(x_train)
        scaledX = scaler.transform(x_train)

##### INITILASE THE VALUES FOR GRID SEARCH FOR VARIOUS PARAMETERS########
 
        k_values = np.array([1,3,5,7,9,11,13,15,17,19,21])
        max_depth=np.array([1,2,3,4,5,6,7,8,9])
        min_samples_leaf=np.array([2,4,6,8,10,12,14])
        param_gridknn = dict(n_neighbors=k_values)
        param_griddt=dict(max_depth=max_depth,min_samples_leaf=min_samples_leaf)

## FINDING THE VARIOUS PARAMTERS FOR DIFFERENT MODELS:
        for name, model in models:
            if name=='KNN':
                kfold = KFold(n_splits=num_folds, random_state=seed)
                grid = GridSearchCV(estimator=model, param_grid=param_gridknn,cv=kfold)
                grid_result = grid.fit(scaledX, y_train)
                print("Best: %f using %s for %s" % (grid_result.best_score_, grid_result.best_params_,name))
            elif name=='CART':
                kfold = KFold(n_splits=num_folds, random_state=seed)
                grid = GridSearchCV(estimator=model, param_grid=param_griddt,cv=kfold)
                grid_result = grid.fit(scaledX, y_train)
                print("Best: %f using %s for %s" % (grid_result.best_score_, grid_result.best_params_,name))
            elif name=='RF':
                kfold = KFold(n_splits=num_folds, random_state=seed)
                grid = GridSearchCV(estimator=model, param_grid=param_griddt,cv=kfold)
                grid_result = grid.fit(scaledX, y_train)
                print("Best: %f using %s for %s" % (grid_result.best_score_, grid_result.best_params_,name))

######## FOR REGRESSION#############################  
 
    elif reg==1:
        models=importregressionmodels()
        results = []
        names = []
        from sklearn.model_selection import KFold
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import accuracy_score
 
        
        num_folds =num_folds
        seed = 77
        f=metrics
        if f=='mse':
            scoring='neg_mean_squared_error'
        elif f=='mae':
            scoring='neg_mean_absolute_error'
        elif f=='acc':
            scoring='r2'
        for name, model in models:
            kfold = KFold(n_splits=num_folds, random_state=seed)
            cv_results = cross_val_score(model, x_train, y_train, cv=kfold,scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
### PARAMETER TUNING GRID SEARCH
 
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(x_train)
        scaledX = scaler.transform(x_train)
    ## INITILASE THE VALUES FOR GRID SEARCH FOR VARIOUS PARAMETERS
        k_values = np.array([1,3,5,7,9,11,13,15,17,19,21])
        max_depth=np.array([1,2,3,4,5,6,7,8,9])
        min_samples_leaf=np.array([2,4,6,8,10,12,14])
        param_gridknn = dict(n_neighbors=k_values)
        param_griddt=dict(max_depth=max_depth,min_samples_leaf=min_samples_leaf)
    ## FINDING THE VARIOUS PARAMTERS FOR DIFFERENT MODELS:
        for name, model in models:
            if name=='KNN':
                kfold = KFold(n_splits=num_folds, random_state=seed)
                grid = GridSearchCV(estimator=model, param_grid=param_gridknn,cv=kfold,scoring=scoring)
                grid_result = grid.fit(scaledX, y_train)
                print("Best: %f using %s for %s" % (grid_result.best_score_, grid_result.best_params_,name))
            elif name=='CART':
                kfold = KFold(n_splits=num_folds, random_state=seed)
                grid = GridSearchCV(estimator=model, param_grid=param_griddt,cv=kfold,scoring=scoring)
                grid_result = grid.fit(scaledX, y_train)
                print("Best: %f using %s for %s" % (grid_result.best_score_, grid_result.best_params_,name))
            elif name=='RF':
                kfold = KFold(n_splits=num_folds, random_state=seed)
                grid = GridSearchCV(estimator=model, param_grid=param_griddt,cv=kfold,scoring=scoring)
                grid_result = grid.fit(scaledX, y_train)
                print("Best: %f using %s for %s" % (grid_result.best_score_, grid_result.best_params_,name))