# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 06:35:28 2016

@author: rob
"""

class StackingAttributes:

    def __init__(self, X_train, Y_train, X_test, Y_test, X_testsub, X_blindsub):
        self._rf_n_estimators = 100
        self._rf_max_depth=175
        self._rf_min_samples_split=9
        self._rf_min_samples_leaf=1
        self._rf_random_state = 1,
        self._rf_n_jobs=-1
        
        self._svc_C = 10
        self._svc_gamma=0.01
        self._svc_kernel='rbf'
        self._svc_probability=True
        self._svc_random_state = 2,
        self._svc_n_jobs=-1       
        
        self._kn_n_neighbors=15
        self._kn_weights='distance'

                                      
        self._X_train = X_train
        self._X_test = X_test
        self._X_testsub = X_testsub
        self._X_blindsub = X_blindsub
        self._Y_train = Y_train
        self._Y_test = Y_test
        
        self._final_pred = None
        self._final_pred_testsub = None
        self._final_pred_blindsub = None
        
        self._gb_n_estimators = 1000
        self._gb_learning_rate = 0.01
        
    @property
    def gb_n_estimators(self):
        """"""
        return self._gb_n_estimators

    @gb_n_estimators.setter
    def gb_n_estimators(self, value):
        self._gb_n_estimators = value
        
    @property
    def gb_learning_rate(self):
        """"""
        return self._gb_learning_rate

    @gb_learning_rate.setter
    def gb_learning_rate(self, value):
        self._gb_learning_rate = value
        
        
    @property
    def rf_n_estimators(self):
        """"""
        return self._rf_n_estimators

    @rf_n_estimators.setter
    def rf_n_estimators(self, value):
        self._rf_n_estimators = value
        
        
        
        
    @property
    def rf_n_estimators(self):
        """"""
        return self._rf_n_estimators

    @rf_n_estimators.setter
    def rf_n_estimators(self, value):
        self._rf_n_estimators = value
        
    @property
    def rf_max_depth(self):
        """"""
        return self._rf_max_depth

    @rf_max_depth.setter
    def rf_max_depth(self, value):
        self._rf_max_depth = value
        
    @property
    def rf_min_samples_split(self):
        """"""
        return self._rf_min_samples_split

    @rf_min_samples_split.setter
    def rf_min_samples_split(self, value):
        self._rf_min_samples_split = value
        
    @property
    def rf_min_samples_leaf(self):
        """"""
        return self._rf_min_samples_leaf

    @rf_min_samples_leaf.setter
    def rf_min_samples_leaf(self, value):
        self._rf_min_samples_leaf = value
        
    @property
    def rf_random_state(self):
        """"""
        return self._rf_random_state

    @rf_random_state.setter
    def rf_random_state(self, value):
        self._rf_random_state = value
        
    @property
    def rf_n_jobs(self):
        """"""
        return self._rf_n_jobs

    @rf_n_jobs.setter
    def rf_n_jobs(self, value):
        self._rf_n_jobs = value
        

    @property
    def svc_C(self):
        """"""
        return self._svc_C

    @svc_C.setter
    def svc_C(self, value):
        self._svc_C = value
        
    @property
    def svc_gamma(self):
        """"""
        return self._svc_gamma

    @svc_gamma.setter
    def svc_gamma(self, value):
        self._svc_gamma = value
        
    @property
    def svc_kernel(self):
        """"""
        return self._svc_kernel

    @svc_kernel.setter
    def svc_kernel(self, value):
        self._svc_kernel = value
        
    @property
    def svc_probability(self):
        """"""
        return self._svc_probability

    @svc_probability.setter
    def svc_probability(self, value):
        self._svc_probability = value
        
        
    @property
    def svc_random_state(self):
        """"""
        return self._svc_random_state

    @svc_random_state.setter
    def svc_random_state(self, value):
        self._svc_random_state = value
        
    @property
    def svc_n_jobs(self):
        """"""
        return self._svc_n_jobs

    @svc_n_jobs.setter
    def svc_n_jobs(self, value):
        self._svc_n_jobs = value
   
 
    @property
    def kn_n_neighbors(self):
        """"""
        return self._kn_n_neighbors

    @kn_n_neighbors.setter
    def kn_n_neighbors(self, value):
        self._kn_n_neighbors = value
        
    @property
    def kn_weights(self):
        """"""
        return self._kn_weights

    @kn_weights.setter
    def kn_weights(self, value):
        self._kn_weights = value
        

        
    @property
    def X_train(self):
        """"""
        return self._X_train

    @X_train.setter
    def X_train(self, value):
        self._X_train = value

    @property
    def X_test(self):
        """"""
        return self._X_test

    @X_test.setter
    def X_test(self, value):
        self._X_test = value
        
    @property
    def X_testsub(self):
        """"""
        return self._X_testsub

    @X_testsub.setter
    def X_testsub(self, value):
        self._X_testsub = value

    @property
    def X_blindsub(self):
        """"""
        return self._X_blindsub

    @X_blindsub.setter
    def X_blindsub(self, value):
        self._X_blindsub = value

        
    @property
    def Y_train(self):
        """"""
        return self._Y_train

    @Y_train.setter
    def Y_train(self, value):
        self._Y_train = value
        
    @property
    def Y_test(self):
        """"""
        return self._Y_test

    @Y_test.setter
    def Y_test(self, value):
        self._Y_test = value
        
    @property
    def final_pred(self):
        """"""
        return self._final_pred

    @final_pred.setter
    def final_pred(self, value):
        self._final_pred = value
        
    @property
    def final_pred_testsub(self):
        """"""
        return self._final_pred_testsub

    @final_pred_testsub.setter
    def final_pred_testsub(self, value):
        self._final_pred_testsub = value
        
    @property
    def final_pred_blindsub(self):
        """"""
        return self._final_pred_blindsub

    @final_pred_blindsub.setter
    def final_pred_blindsub(self, value):
        self._final_pred_blindsub = value
        
        