# -*- coding: utf-8 -*-

# requires sciket-learn 0.18
# if required, conda update scikit-learn
import math
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV   #Performing grid search
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier


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
    
    # Normalise the data attributes
    X = (X - X.mean(axis=0)) /  X.std(axis=0)
    
    return (X)
    

def extractFeatures( X, nFeatures, ranking ):
    importance = pd.DataFrame({'feature': X.columns, 'rank': ranking})
    importance.sort_values(['rank', 'feature'], ascending=[1, 1], inplace=True)
    
    deleteCols = importance.feature[nFeatures:]
    
    return X.drop(X.columns[deleteCols],axis=1)
    
    
def featureSelection( X ):

    clfName = 'rf'
    X_features = []

    class1_nFeatures = 158
    class1_ranking = np.load('{}_class1_ranking.npy'.format(clfName))

    class2_nFeatures = 147
    class2_ranking = np.load('{}_class2_ranking.npy'.format(clfName))
    
    class3_nFeatures = 153
    class3_ranking = np.load('{}_class3_ranking.npy'.format(clfName))
    
    class4_nFeatures = 64
    class4_ranking = np.load('{}_class4_ranking.npy'.format(clfName))
 
    X_features.append(extractFeatures( X, class1_nFeatures, class1_ranking ))
    X_features.append(extractFeatures( X, class2_nFeatures, class2_ranking ))
    X_features.append(extractFeatures( X, class3_nFeatures, class3_ranking ))

    X4 = extractFeatures( X, class4_nFeatures, class4_ranking )
    
    print(np.any(np.isnan(X4)))


    print(np.all(np.isfinite(X4)))

    print(X4.describe())
    X4 = X4.apply(np.log)
    X_features.append(X4)
    
    return X_features

def runModel(X_train, X_test, Y_train, Y_test):
    """ Create model and run predictions
    """
    
    numberClasses = 4
    
    Y_predict = np.zeros((Y_test[0].shape[0], numberClasses), dtype=np.float)
    models = []

    
    for i in range(numberClasses):
        print('Model: ', i+1)
#        model = RandomForestClassifier(
#                n_estimators=300, 
#                max_depth=180, 
#                min_samples_split=6, 
#                min_samples_leaf=1,
#                random_state=24,   
#                oob_score=True)        
        model = QuadraticDiscriminantAnalysis()
        model.fit(X_train[i],Y_train[i])
        Y_predict[:,i] = model.predict_proba(X_test[i])[:,1]
        models.append(model)
    
    return Y_predict, models


def runEnsembleModel(X_train, X_test, Y_train, Y_test):
    print('Create model')
    
      
    numberClasses = 4
    
    Y_predict = np.zeros((Y_test[0].shape[0], numberClasses), dtype=np.float)
    Model1_predict_train = np.zeros((Y_test[0].shape[0], numberClasses), dtype=np.float)
    Model5_predict_train = np.zeros((Y_test[0].shape[0], numberClasses), dtype=np.float)
    
    models = []

    for i in range(numberClasses):
        print('Model: ', i+1)
        Model1 = RandomForestClassifier(
                n_estimators=150, 
                max_depth=128, 
                min_samples_split=4, 
                min_samples_leaf=2,
                random_state=24,   
                oob_score=False)  
        Model1.fit(X_train[i],Y_train[i])
        Model1_predict_train = Model1.predict_proba(X_train[i])[:,1]
        Model1_predict_test = Model1.predict_proba(X_test[i])[:,1]
        
        # Build Model3 - Level 0
        Model3 = (QuadraticDiscriminantAnalysis())
        Model3.fit(X_train[i],Y_train[i])
        Model3_predict_train = Model3.predict_proba(X_train[i])[:,1]
        Model3_predict_test = Model3.predict_proba(X_test[i])[:,1]
    
        Model5 = KNeighborsClassifier(n_neighbors=15, weights='distance')               
        Model5.fit(X_train[i],Y_train[i])
        Model5_predict_train = Model5.predict_proba(X_train[i])[:,1]
        Model5_predict_test = Model5.predict_proba(X_test[i])[:,1]
        
        #Model 4 - Level 1 
        #Creating training attributes for Model4 (based on Model1, Model3, Model5 )
        FeaturesTrain1 = np.array([Model1_predict_train, Model3_predict_train, Model5_predict_train]) 
        
        #print('Feature ', FeaturesTrain1.T.shape)
        #print('YTrain ', Y_train[i].shape)
        
        FinalModel = LogisticRegression(random_state=49)
        FinalModel.fit(FeaturesTrain1.T,Y_train[i])
    
        FeaturesTest1 = np.array([Model1_predict_test, Model3_predict_test, Model5_predict_test]) 
            
        #Final predictions
        Y_predict[:,i] = FinalModel.predict_proba(FeaturesTest1.T)[:,1]
    
        models.append(FinalModel)
    
    return Y_predict, models
    

def calculateROC(y_bin, Y_predict):
    """ Calculate area under ROC curve
    """
    print('Calc ROC')
    # Binarize the output
    #y_bin = label_binarize(Y, classes=[1, 2, 3,4])

    #Calculate AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_bin[i], Y_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return roc_auc

def createSubmission(models, preprocessData=preprocessFeatures3):
    print('Create Submission')
    #Create submission
    Xtest = pd.read_csv("testData.txt",sep="\t",header=None)
    
    m,n = Xtest.shape
    Xtest = preprocessData(Xtest)
    Xtest = featureSelection(Xtest)
    
    y_final_prob = np.zeros((m, 4), dtype=np.float)
    #y_final_predict = np.zeros((m, 4), dtype=np.float)
    y_final_label = np.zeros((m, 1), dtype=np.float)
    
    for i in range(len(models)):
        y_final_prob[:,i] = models[i].predict_proba(Xtest[i])[:,1]
        #y_final_predict[:,i] = models[i].predict(Xtest) 

    # Convert back to a class
    y_final_label = np.argmax(y_final_prob, axis=1)
    y_final_label += 1
    
        
    sample = pd.DataFrame(np.hstack([y_final_prob.round(5),y_final_label.reshape(y_final_prob.shape[0],1)]))
    sample.columns = ["prob1","prob2","prob3","prob4","label"]
    sample.label = sample.label.astype(int)
    
    #Submit this file to dropbox
    sample.to_csv("Johnston_Memic.csv",sep="\t" ,index=False,header=None)
    
    
def trainTestSplit( X, Y ):
    
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    
    for i in range(len(X)):    
        xtrain, xtest, ytrain, ytest = train_test_split(X[i], Y[:,i], test_size=.33, random_state=10) 
        X_train.append(xtrain)
        X_test.append(xtest)
        Y_train.append(ytrain)
        Y_test.append(ytest)
        
    return X_train, X_test, Y_train, Y_test
    
    
def main():

    # Read the files in.   
    (XOrig,YOrig) = readFiles()
    
    #class1_scores = pd.read_csv("class1_scores.txt",sep='\s+',header=None)
    #print(class1_scores)

    # Clean up the data
    X = preprocessFeatures3(XOrig)
    X = featureSelection(X)
    
    Y = label_binarize(YOrig, classes=[1, 2, 3, 4])
    

    # Split into training and test set  
    X_train, X_test, Y_train, Y_test = trainTestSplit(X, Y)
    
    
    # Run the model
    Y_predict, model = runModel(X_train, X_test, Y_train, Y_test)
    #Y_predict, model = runEnsembleModel(X_train, X_test, Y_train, Y_test)
    

    
    AUC = calculateROC(Y_test, Y_predict)
    
    print('AUC is: ', AUC)
    
    createSubmission(model)

    print('Completed')
 
if __name__ == '__main__':
  main()


# Rob's random forest:
# AUC is:  {0: 0.93190766680705939, 1: 0.79866246656166395, 2: 0.92063532074118881, 3: 0.71576178069777985}

# Haris's random forest:
# AUC is:  {0: 0.94306685868782525, 1: 0.82743235247547853, 2: 0.93536233223544174, 3: 0.74154621658359765}

# Haris's 2nd random forest:
# AUC is:  {0: 0.95000475524095485, 1: 0.83333666675000206, 2: 0.94117912717617802, 3: 0.74163797009515187}

# Only 64 features:
# AUC is:  {0: 0.86416722144477809, 1: 0.77907406018483805, 2: 0.81123966525509184, 3: 0.7652242863615768}
# AUC is:  {0: 0.94940525522057528, 1: 0.77907406018483805, 2: 0.81123966525509184, 3: 0.7652242863615768}
# AUC is:  {0: 0.9498961339216897, 1: 0.79040950366465379, 2: 0.80503624995482315, 3: 0.75502398589065267}
# AUC is:  {0: 0.9498961339216897, 1: 0.79040950366465379, 2: 0.94087028804799611, 3: 0.75502398589065267}
# AUC is:  {0: 0.9498961339216897, 1: 0.84989296352265375, 2: 0.94087028804799611, 3: 0.75502398589065267}

# QDA
# AUC is:  {0: 0.97986879722719467, 1: 0.85936029100498579, 2: 0.95235476526076124, 3: 0.7551148736037625}

