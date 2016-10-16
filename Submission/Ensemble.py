# -*- coding: utf-8 -*-
import numpy as np



from sklearn.metrics import roc_curve, auc

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC # "Support Vector Classifier"
from sklearn.feature_selection import RFE

class Ensemble:
    
    def __init__(self, attrs):
        self.attrs = attrs
        
    def run(self, verbose):
        if verbose > 0:
            print("Running ensemble")

        #Build Model1 - Level 0    
        if verbose > 0:
            print("Running Random Forest Classifier")
            
        Model1 = RandomForestClassifier(
            n_estimators = self.attrs.rf_n_estimators,
            max_depth = self.attrs.rf_max_depth,
            min_samples_split = self.attrs.rf_min_samples_split,
            min_samples_leaf = self.attrs.rf_min_samples_leaf,
            random_state = 1,
            n_jobs = self.attrs.rf_n_jobs)
            
        if self.attrs.rf_use_rfe:
            if verbose > 0:
                print(" using RFE")
            Model1 = RFE(Model1, n_features_to_select = 150, step = 20)
            
        Model1.fit(self.attrs.X_train, self.attrs.Y_train)

        #Predict on X_train, X_test
        Model1_pred_test = Model1.predict_proba(self.attrs.X_test)
        Model1_pred_train = Model1.predict_proba(self.attrs.X_train)
        Model1_pred_testsub = Model1.predict_proba(self.attrs.X_testsub)
        Model1_pred_blindsub = Model1.predict_proba(self.attrs.X_blindsub)

        #Build Model2 - Level 0
        if verbose > 0:
            print("Running SVM Classifier")

        Model2 = SVC(C = self.attrs.svc_C, 
                     gamma = self.attrs.svc_gamma, 
                     kernel= self.attrs.svc_kernel, 
                     probability = self.attrs.svc_probability,
                     random_state = 2)
        Model2.fit(self.attrs.X_train, self.attrs.Y_train)
        #Predict on X_train, X_test
        Model2_pred_test = Model2.predict_proba(self.attrs.X_test)
        Model2_pred_train = Model2.predict_proba(self.attrs.X_train)
        Model2_pred_testsub = Model2.predict_proba(self.attrs.X_testsub)
        Model2_pred_blindsub = Model2.predict_proba(self.attrs.X_blindsub)

        #Build Model3 - Level 0
        if verbose > 0:
            print("Quadratic Discriminant Analysis Classifier")

        Model3 = QuadraticDiscriminantAnalysis()
        Model3.fit(self.attrs.X_train, self.attrs.Y_train)
        #Predict on X_train, X_test
        Model3_pred_test = Model3.predict_proba(self.attrs.X_test)
        Model3_pred_train = Model3.predict_proba(self.attrs.X_train)
        Model3_pred_testsub = Model3.predict_proba(self.attrs.X_testsub)
        Model3_pred_blindsub = Model3.predict_proba(self.attrs.X_blindsub)

        #Build Model4 - Level 0
        if verbose > 0:
            print("GaussianNB Classifier")

        Model4 = GaussianNB()
        Model4.fit(self.attrs.X_train, self.attrs.Y_train)
        #Predict on X_train, X_test
        Model4_pred_test = Model4.predict_proba(self.attrs.X_test)
        Model4_pred_train = Model4.predict_proba(self.attrs.X_train)
        Model4_pred_testsub = Model4.predict_proba(self.attrs.X_testsub)
        Model4_pred_blindsub = Model4.predict_proba(self.attrs.X_blindsub)

        #Build Model5 - Level 0
        if verbose > 0:
            print("KNeighbors Classifier")

        Model5 = KNeighborsClassifier(n_neighbors = self.attrs.kn_n_neighbors, 
                                      weights = self.attrs.kn_weights)
                            
        Model5.fit(self.attrs.X_train, self.attrs.Y_train)
        #Predict on X_train, X_test
        Model5_pred_test = Model5.predict_proba(self.attrs.X_test)
        Model5_pred_train = Model5.predict_proba(self.attrs.X_train)
        Model5_pred_testsub = Model5.predict_proba(self.attrs.X_testsub)
        Model5_pred_blindsub = Model5.predict_proba(self.attrs.X_blindsub)

        #Build Model6 - Level 0
        if verbose > 0:
            print("Logistic Regression Classifier")

        Model6 = LogisticRegression(C = self.attrs.lr_C, 
                                      random_state = 6)
                            
        Model6.fit(self.attrs.X_train, self.attrs.Y_train)
        #Predict on X_train, X_test
        Model6_pred_test = Model6.predict_proba(self.attrs.X_test)
        Model6_pred_train = Model6.predict_proba(self.attrs.X_train)
        Model6_pred_testsub = Model6.predict_proba(self.attrs.X_testsub)
        Model6_pred_blindsub = Model6.predict_proba(self.attrs.X_blindsub)
        
        #Final Model - Level 1 
        #Creating training attributes for the stacked model
        if verbose > 0:
            print("Stacked Classifier")
            
        FeaturesTrain1 = np.hstack([Model1_pred_train,
                                    Model2_pred_train,
                                    Model3_pred_train,
                                    Model4_pred_train,
                                    Model5_pred_train,
                                    Model6_pred_train])  
        ModelFinal = LogisticRegression(random_state=49)
        ModelFinal.fit(FeaturesTrain1, self.attrs.Y_train)

        # Save the final model in case we want to work with it later
        self.attrs.final_model = ModelFinal

        #Creating test attributes final model
        Features_test1 = np.hstack([Model1_pred_test,
                                    Model2_pred_test,
                                    Model3_pred_test,
                                    Model4_pred_test,
                                    Model5_pred_test,
                                    Model6_pred_test])
        Features_testsub1 = np.hstack([Model1_pred_testsub,
                                       Model2_pred_testsub,
                                       Model3_pred_testsub,
                                       Model4_pred_testsub,
                                       Model5_pred_testsub,
                                       Model6_pred_testsub])
        Features_blindsub1 = np.hstack([Model1_pred_blindsub,
                                        Model2_pred_blindsub,
                                        Model3_pred_blindsub,
                                        Model4_pred_blindsub,
                                        Model5_pred_blindsub,
                                        Model6_pred_blindsub])


        #Final predictions
        self.attrs.final_pred = ModelFinal.predict_proba(Features_test1)
        self.attrs.final_pred_testsub = ModelFinal.predict_proba(Features_testsub1)
        self.attrs.final_pred_blindsub = ModelFinal.predict_proba(Features_blindsub1)

        #AUC
        if verbose > 0:
            print("Calculating AUC")
            
        fpr, tpr, thresholds = roc_curve(self.attrs.Y_test, self.attrs.final_pred[:, 1])
        roc_auc = auc(fpr, tpr)
        print("AUC with Stacking: " , roc_auc)



