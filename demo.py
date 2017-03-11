# -*- coding: utf-8 -*-
"""
Created on Sun Oct 05 23:58:46 2016

@author: huanghe
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pickle
#from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsOneClassifier
import time
from sklearn import metrics
def listDown(s):
    t = []
    for i in np.arange(len(s)):
        t.append(' '.join(s[i]))
    return t

    
def find_best_words(word_scores, number):    
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, \
                reverse=True)[:number] #把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的    
    best_words = set([w for w, s in best_vals])    
    return best_words
def calculate_result(actual,pred):
    m_precision=metrics.precision_score(actual,pred)
#    m_recall=metrics.recall_score(actual,pred)
    print 'precision:{0:.3f}'.format(m_precision)  
#    print 'recall:{0:0.3f}'.format(m_recall);  
#    print 'f1-score:{0:.3f}'.format(metrics.f1_score(actual,pred));  
feature_number = 20000
#=================load the data=======================
start_time=time.time()
label = 'gender'    # 'age','gender','education'	
trainData_all = np.load('train_unigram_11_6.npy')
trainLabel_all = pd.read_csv('trainLabel.csv',sep=' ')
trainLabel_all = np.array((trainLabel_all[label]))
#remove the unknown data 
known = [i for i in range(len(trainLabel_all)) if trainLabel_all[i]!=0]
trainData_ = trainData_all[known]
trainLabel = trainLabel_all[known]
dataload_time=time.time()
#print "load time=",dataload_time-start_time
#===============load the dictionary===================

f = open('word_scores_'+label+'_20000.txt')
word_scores = pickle.load(f)
f.close()
si_keys = list(find_best_words(word_scores,feature_number))  # a list of most informative words
keys = list(set(si_keys))
values = range(len(keys))
dict_fromChi2 = dict(zip(keys, values))
dictload_time=time.time()
print "dict load=",dictload_time-dataload_time
#==============TFIDF feature extraction====================

tfv = TfidfVectorizer(min_df=3,max_df=0.5,vocabulary = dict_fromChi2)
trainData = tfv.fit_transform(listDown(trainData_))
tfidf_time=time.time()
#print "tfidf time=",tfidf_time-dictload_time
'''-----------------classify--------------------'''
trainData_part=trainData[:18000]
trainLabel_part=trainLabel[:18000]
testData_dev=trainData[18000:]
testLabel_dev=trainLabel[18000:]

'''-------------------SVM--------------'''
from sklearn.svm import SVC
print '*************************\nSVM\n*************************'  
svclf=SVC(kernel='linear')
svclf.fit(trainData_part,trainLabel_part)
pred =svclf.predict(testData_dev)
calculate_result(testLabel_dev,pred)

#'''---------------Multinomial Naive bayes -----'''
#from sklearn.naive_bayes import  MultinomialNB
#print '*************************\nNaive Bayes\n*************************'
#clf=MultinomialNB(alpha=0.01)
#clf.fit(trainData_part,trainLabel_part)
#pred=clf.predict(testData_dev)
#print "Multinomial Naive Bayes"
#calculate_result(testLabel_dev,pred)
#
#'''--------------KNN-------'''
#from sklearn.naive_bayes import  KNeighborsClassifier
#print '*************************\nKNN\n*************************' 
#knnclf = KNeighborsClassifier()#default with k=5  
#knnclf.fit(trainData_part,trainLabel_part)  
#pred = knnclf.predict(testData_dev);  
#calculate_result(testLabel_dev,pred); 
#
#'''---------------------KMeans--------------------'''
#from sklearn.cluster import KMeans  
#print '*************************\nKMeans\n*************************'  
#pred = KMeans(n_clusters=5)  
#pred.fit(testData_dev)  
#calculate_result(testLabel_dev,pred.labels_);  
##==================four classifiers========================
## NB, best parameters:  gender: 1e-8
#clf_NB = BernoulliNB(alpha=1e-8)
#temp = cross_val_score(clf_NB,trainData,trainLabel, cv=10)
#print 'BernoulliNB'  
#print temp
#print temp.mean()
#
## MNB, best parameters: gender: 1e-8 age:1e-3 education:1e-2 
#clf_MNB = MultinomialNB(alpha=1e-2)
#temp = cross_val_score(clf_MNB,trainData,trainLabel, cv=10)
#print '\nMultinomialNB'
#print temp
#print temp.mean()
#
## MaxEnt, parameters untuned  gender:1.5 
#from sklearn.linear_model import LogisticRegression
#clf_ME = LogisticRegression(n_jobs=-1)
#temp = cross_val_score(clf_ME,trainData,trainLabel, cv=10)
#print '\nMaxEnt'
#print temp
#print temp.mean()
#
## SVM, transform into probability output
#from sklearn.calibration import CalibratedClassifierCV
#clf_ = svm.LinearSVC(C=0.2) #still the best when C=0.2
##clf_ = OneVsOneClassifier(svm.LinearSVC(random_state=0,C=0.2))
#clf_SVM = CalibratedClassifierCV(clf_)
#temp = cross_val_score(clf_SVM,trainData,trainLabel, cv=10)
#print '\nSVM'
#print temp
#print temp.mean()
#

##==========================ensemble===========================
#from sklearn.ensemble import VotingClassifier
#clf_ensemble = VotingClassifier(estimators=[('NB', clf_NB), ('MNB', clf_MNB), ('ME', clf_ME),\
#                                     ('SVM',clf_SVM)], voting='soft')
#temp = cross_val_score(clf_ensemble,trainData,trainLabel, cv=5)
#print '\nEnsemble'
#print temp
#print temp.mean()
#
## predict
#clf_ensemble.fit(trainData,trainLabel)
#testData_ = np.load('testData_afterClean_3.npy')
#testData = tfv.transform(listDown(testData_))
#pred = clf_ensemble.predict(testData)
#classify_time=time.time()
#print "classify time=",classify_time-tfidf_time
##=========save the result for age/gender/education respectively=====
#submission = pd.read_csv('submission_eighth.csv',sep=' ')
#submission[label] = pred
#submission.to_csv('submission_eighth.csv',sep=' ',index=False)
#end_time=time.time()
#print "built csv time=",end_time-classify_time
#print "all time=",end_time-start_time
#
