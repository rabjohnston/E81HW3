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
from sklearn.preprocessing import MinMaxScaler

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
    

def extractFeatures( X, nFeatures, ranking ):
    importance = pd.DataFrame({'feature': X.columns, 'rank': ranking})
    importance.sort_values(['rank', 'feature'], ascending=[1, 1], inplace=True)
    
    deleteCols = importance.feature[nFeatures:]
    print(deleteCols)
    
    return X.drop(X.columns[deleteCols],axis=1)
    
    
def featureSelection( X ):

    #X_features = np.zeros((X.shape[0],4), dtype=np.float)
    X_features = []

    class1_nFeatures = 158
    class1_ranking = [110,   1,  24,  54,   1,   1,   1, 160,   1,  44,   1,  64, 120, 101,   1,   1, 177,   1,
   1,  61, 127,   1, 147,   1,   1, 119,  22,   2,   1,   1,  47,   4, 145, 138,  83,   1,
 126,  94,  35,   1, 129,   1,   1,  23,  86,   1,   1,   1,   1,   1,  66,   1, 122,   9,
 104,  90, 134, 128,   1,   1,  27, 168,   5,  80,   1, 167,   1, 174, 117,   1,  26,   1,
  31,  48,   1,   1, 149,   1,   1,   1,  52,  79, 112,  95,   1,  42,   1,   1,   1,   1,
  81,   1,   1,  91,  89,  29,   1, 170,   1,   1,   1,   1,  15,  97, 140, 107,   1, 102,
   1,  11, 146, 113, 108, 109,   1, 152,   1,  17,   1,   1,  73,   1,  40,   1,  43,   1,
 151,   1,   1,   1, 105,  63,   1,   1,   1, 144,   1,   1,   1,   1,   8,   1,   1, 103,
 125, 131,   1,   1,   1,  78,  51,  82, 164,   1,   1,   1,   1,   1, 165, 121,   1, 159,
   1,   1,  37,  21,   1,   1,  93,  38,   1, 153,  14, 130,   1, 172,   1, 136,   1,   7,
  84, 123,   1,  71,  34, 141, 135,  92,  18, 124,  74, 143,   1,   1,   1,   1,  39,   1,
   1,   1,   1,  87,   1,   1,  19,  60, 158,   1,   1,   1,  46,   1,   1,   1, 157,   1,
   1,  13,  69,  70,   1,   1,  62,  16, 114,  68, 111,  55,   1,   1,   1,   1,   1,   1,
   1,   1,  76, 175,   1,  53,  10,   1,   1,   1,   1,  50,  58,   1,   1,   1,  25, 106,
  49, 139,  65,  56,  96,   1,   1,   1,   1,   1,  41,  98,   1, 163,  72,  20,  1 , 12,
 132, 166, 169,   1,   1,  67,  85,  36, 173,   1,   1, 116,   1,  57,  88, 162,  1 , 75,
  30,   1,   3,   1,   1,   1,   1, 156,   1, 100,   1,   1, 171, 161,  99,   1, 155,   1,
 142,   1,  32,   1, 150, 148,   1,  28,  77,   1, 115,   1, 137,   1,   6, 133,   1, 176,
   1,   1, 118,  59,   1, 154,  33,   1,  45,   1]

    class4_nFeatures = 64
    class4_ranking = \
    [175, 73, 39, 86, 11, 1, 24, 201, 1, 3, 33, 208, 192, 166, 30, 68, 212, 1, \
    91, 1, 119, 164, 61, 1, 1, 263, 184, 248, 83, 21, 1, 116, 182, 113, 146, 1, \
    118, 225, 1, 261, 193, 1, 162, 48, 180, 37, 264, 36, 202, 1, 229, 70, 271, 17, \
    85, 239, 152, 240, 5, 1, 206, 140, 51, 4, 120, 265, 69, 131, 154, 1, 1, 104, \
    34, 49, 1, 1, 244, 57, 45, 138, 115, 233, 253, 10, 8, 260, 62, 50, 98, 29, \
    81, 16, 189, 153, 114, 76, 1, 41, 1, 20, 179, 1, 12, 129, 128, 177, 149, 246, \
    1, 54, 236, 1, 92, 72, 35, 64, 268, 1, 127, 1, 32, 1, 2, 22, 203, 15, \
    42, 106, 9, 1, 6, 71, 174, 173, 79, 109, 200, 142, 169, 121, 1, 55, 144, 227, \
    168, 1, 122, 90, 1, 1, 60, 75, 157, 96, 1, 155, 150, 7, 221, 249, 216, 159, \
    1, 80, 250, 218, 220, 1, 102, 170, 1, 194, 195, 197, 1, 59, 181, 1, 1, 27, \
    230, 237, 262, 1, 226, 238, 269, 231, 1, 137, 205, 255, 1, 143, 107, 44, 1, 1, \
    130, 43, 78, 31, 1, 124, 1, 77, 171, 97, 18, 190, 1, 1, 1, 95, 223, 111, \
    258, 163, 172, 211, 26, 185, 82, 147, 191, 84, 252, 89, 28, 13, 151, 103, 1, 145, \
    1, 1, 110, 266, 204, 161, 67, 101, 178, 1, 188, 207, 87, 1, 58, 1, 214, 196, \
    126, 117, 259, 257, 219, 105, 88, 108, 1, 1, 23, 198, 139, 14, 234, 245, 1, 209,
    241, 141, 213, 47, 52, 132,  53, 167, 228,   1,   1, 254, 133, 199,  65,   1, 165, 156, \
    1,  94, 123,  74,  40,  99, 100, 247, 222,  38, 256,   1,   1, 125, 134,  19, 210,  56, \
    93, 160, 158, 186, 215, 176,   1, 267, 136, 112, 224,  46,  66, 187,   1, 232, 183, 217, \
    135,  63, 243, 251, 270, 242,  25, 148, 235,   1]
 
    X_features.append(extractFeatures( X, class1_nFeatures, class1_ranking ))
    X_features.append(extractFeatures( X, class4_nFeatures, class4_ranking ))
    X_features.append(extractFeatures( X, class4_nFeatures, class4_ranking ))
    X_features.append(extractFeatures( X, class4_nFeatures, class4_ranking ))
    
    return X_features

def runModel(X_train, X_test, Y_train, Y_test):
    """ Create model and run predictions
    """
    
    numberClasses = 4
    
    Y_predict = np.zeros((Y_test[0].shape[0], numberClasses), dtype=np.float)
    models = []

    
    for i in range(numberClasses):
        print('Model: ', i+1)
        model = RandomForestClassifier(
                n_estimators=300, 
                max_depth=180, 
                min_samples_split=6, 
                min_samples_leaf=1,
                random_state=24,   
                oob_score=False)     
   
   
        model.fit(X_train[i],Y_train[i])
        a = model.predict_proba(X_test[i])[:,1]
        Y_predict[:,i] = a
        models.append(model)
    
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
    
    
def trainTestSplit( X, Y ):
    
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    
    for i in range(len(X)):    
        xtrain, xtest, ytrain, ytest = train_test_split(X[i], Y[:,i], test_size=.15, random_state=10) 
        X_train.append(xtrain)
        X_test.append(xtest)
        Y_train.append(ytrain)
        Y_test.append(ytest)
        
    return X_train, X_test, Y_train, Y_test
    
    
def main():

    # Read the files in.   
    (XOrig,YOrig) = readFiles()
    
    
    # Clean up the data
    X = preprocessFeatures3(XOrig)
    X = featureSelection(X)
    
    Y = label_binarize(YOrig, classes=[1, 2, 3, 4])
    

    #Split into training and test set - where the latter is 15% of the total 
    X_train, X_test, Y_train, Y_test = trainTestSplit(X, Y)
    
    
    # Run the model
    Y_predict, model = runModel(X_train, X_test, Y_train, Y_test)
    
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
