# -*- coding: utf-8 -*-

# requires sciket-learn 0.18
# if required, conda update scikit-learn

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV   #Performing grid search
from sklearn.cross_validation import train_test_split

def readFiles():
    #Reading files
    X = pd.read_csv("trainingData.txt",sep='\t',header=None)
    Y = pd.read_csv("trainingTruth.txt",sep='\t',header=None)
    Y = np.array(Y).ravel()
    
    return (X,Y)


def preprocessFeatures3( X ):
    print('Preprocessing data (3).')

    # Q: Normalise data for SVMs - what about decision trees?
    

    #Rewrite this from the HW ipython book
    # Replace any NaN in X with the mean of the column
    # Replacing with the mean gives a better score
    xMean = []
    for col in X.columns:
        xMean = X[col].mean()
        #print(col, ' ', xMean)
        X.loc[X[col].isnull(), col] = xMean
    
    return (X)



def runModel(X_train, X_test, Y_train, Y_test):
    print('Run model')
    
      
    Y_predict = np.zeros(Y_test.shape, dtype=np.float)
    models = []
    print(Y_train.shape)
    for i in range(Y_train.shape[1]):
        print('Random Forest: ', i)
        clf = RandomForestClassifier(
                n_estimators=50, 
                max_depth=32, 
                min_samples_split=4, 
                min_samples_leaf=4,
                random_state=24,   
                oob_score=False)
        clf.fit(X_train,Y_train[:,i])
        Y_predict[:,i] = clf.predict_proba(X_test)[:,1]
        models.append(clf)
    print( Y_predict)
    
    return Y_predict, models


def calculateROC(y_bin, Y_predict):
    print('Calc ROC')
    # Binarize the output
    #y_bin = label_binarize(Y, classes=[1, 2, 3,4])

    #Calculate AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], Y_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return roc_auc

def createSubmission(models, preprocessData=preprocessFeatures3):
    print('Create Submission')
    #Create submission
    Xtest = pd.read_csv("testData.txt",sep="\t",header=None)
    
    (Xtest) = preprocessData(Xtest)
    
    m,n = Xtest.shape
    y_final_prob = np.zeros((m, 4), dtype=np.float)
    #y_final_predict = np.zeros((m, 4), dtype=np.float)
    y_final_label = np.zeros((m, 1), dtype=np.float)
    
    for i in range(len(models)):
        y_final_prob[:,i] = models[i].predict_proba(Xtest)[:,1]
        #y_final_predict[:,i] = models[i].predict(Xtest) 

    # Convert back to a class
    y_final_label = np.argmax(y_final_prob, axis=1)
    y_final_label += 1
    
        
    sample = pd.DataFrame(np.hstack([y_final_prob.round(5),y_final_label.reshape(y_final_prob.shape[0],1)]))
    sample.columns = ["prob1","prob2","prob3","prob4","label"]
    sample.label = sample.label.astype(int)
    
    #Submit this file to dropbox
    sample.to_csv("Johnston_Memic.csv",sep="\t" ,index=False,header=None)
    

def main():

    # Read the files in.   
    (XOrig,YOrig) = readFiles()
    
    
    # Clean up the data
    X = preprocessFeatures3(XOrig)
    Y = label_binarize(YOrig, classes=[1, 2, 3, 4])
    
    #Split into training and test set - where the latter is 15% of the total 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.15, random_state=10) 
    
    
    # Run the model
    Y_predict, model = runModel(X_train, X_test, Y_train, Y_test)
    
    AUC = calculateROC(Y_test, Y_predict)
    
    print('AUC is: ', AUC)
    
    #createSubmission(model)

    print('Completed')
 
if __name__ == '__main__':
  main()


