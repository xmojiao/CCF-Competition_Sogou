# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 15:58:29 2016

@author: Jiao
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pickle
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
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
    c=0.000
    d=0.000
    for i in range(len(actual)):
        if actual[i]==pred[i]:
            c+=1
    d=float(c)/len(actual)
#    print 'right:{0:.3f}'.format(d) 
    return d

#=================load the data=======================
feature_number = 10000
start_time=time.time()
label = 'age'    # 'age','gender','education'	
#train_bigram=np.load('train_bigram_11_6.npy')
train_unigram = np.load('train_unigram_12_6.npy')
#trainData_all_A=[train_unigram[i]+train_bigram[i] for i in range(len(train_unigram))]
trainData_all_A=[train_unigram[i] for i in range(len(train_unigram))]
trainLabel_all_A = pd.read_csv('trainLabel.csv',sep=' ')
trainLabel_all=trainLabel_all_A[:18000]
trainData_all=trainData_all_A[:18000]
trainLabel_test=trainLabel_all_A[18000:20000]
trainData_test=trainData_all_A[18000:20000]
trainLabel_test = np.array((trainLabel_test[label]))
trainData_test=np.array(trainData_test)

trainLabel_all = np.array((trainLabel_all[label]))
#remove the unknown data 
known = [i for i in range(len(trainLabel_all)) if trainLabel_all[i]!=0]
trainData_all=np.array(trainData_all)
trainData_ = trainData_all[known]
a=trainData_
trainLabel = trainLabel_all[known]
dataload_time=time.time()
#print "load time=",dataload_time-start_time
#===============load the dictionary===================

f = open('word_scores_'+label+'_18000.txt')
word_scores = pickle.load(f)
f.close()
si_keys = list(find_best_words(word_scores,feature_number))  # a list of most informative words
keys = list(set(si_keys))
values = range(len(keys))
dict_fromChi2 = dict(zip(keys, values))
dictload_time=time.time()
#print "dict load=",dictload_time-dataload_time
#==============TFIDF feature extrac.tion====================

tfv = TfidfVectorizer(min_df=3,max_df=0.5,vocabulary = dict_fromChi2)#,stop_words=stop_words)
trainData = tfv.fit_transform(listDown(trainData_))
tfidf_time=time.time()

'''==================four classifiers process========================'''
print '\n---------------',label,'----------------'
print "feature number=",feature_number
trainData_test_ = tfv.transform(listDown(trainData_test))

##''' NB, best parameters:  gender: 1e-8'''
#alpha_NB=2e-3
#clf_NB = BernoulliNB(alpha_NB)   #gender 2e-3  age 3  education 6e-5
#clf_NB.fit(trainData,trainLabel)
#pred = clf_NB.predict(trainData_test_)
#z0=calculate_result(trainLabel_test,pred) 
#print 'alpha_NB=',alpha_NB,'  BernoulliNB=',z0
#
#''' MNB, best parameters: gender: 1e-8 age:1e-3 education:1e-2 '''
#alpha_MNB=4e-2
#clf_MNB = MultinomialNB(alpha_MNB)    #gender  4e-2 age 0.09
#clf_MNB.fit(trainData,trainLabel)
#pred = clf_MNB.predict(trainData_test_)
#z1=calculate_result(trainLabel_test,pred) 
#print 'alpha_MNB=',alpha_MNB,'  MultinomialNB=',z1
#
#''' MaxEnt, parameters untuned  gender:1.5---''' 
#from sklearn.linear_model import LogisticRegression
#clf_ME = LogisticRegression(n_jobs=-1) #gender 不变
#clf_ME.fit(trainData,trainLabel)
#pred = clf_ME.predict(trainData_test_)
#z2=calculate_result(trainLabel_test,pred)
#print ' MaxEnt=', z2#'n_jobs=',n_jobs,

#'''-- SVM, transform into probability output--'''
#from sklearn.calibration import CalibratedClassifierCV
#C=0.15
#clf_ = svm.LinearSVC(C=0.15) #gender c=0.15   age 0.1
#clf_SVM = CalibratedClassifierCV(clf_)
#'''clf_ = OneVsOneClassifier(svm.LinearSVC(random_state=0,C=0.2))'''
#clf_SVM.fit(trainData,trainLabel)
#pred = clf_SVM.predict(trainData_test_)
#z3=calculate_result(trainLabel_test,pred) 
#print 'C=',C,'  SVM=',z3

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100,min_samples_split=2,random_state=0)
clf.fit(trainData,trainLabel)
pred = clf.predict(trainData_test_)
z3=calculate_result(trainLabel_test,pred)
print 'RandomForst',z3
#print "Mean classify: ",(z0+z1+z2+z3)/4
#classify_time=time.time()
#print "classify time=",classify_time-tfidf_time






'''单个算法调参后，使用votingClassifier投票'''
#'''==========================ensemble==========================='''
#from sklearn.ensemble import VotingClassifier
#clf_ensemble = VotingClassifier(estimators=[('NB', clf_NB), ('MNB', clf_MNB), ('ME', clf_ME),\
#                                     ('SVM',clf_SVM)], voting='soft')
#clf_ensemble.fit(trainData,trainLabel)
#pred = clf_ensemble.predict(trainData_test_)
##print "Mean classify: ",(z0+z1+z2+z3)/4
#print "ensemble right: ",calculate_result(trainLabel_test,pred)
#
#classify_time=time.time()
#print "classify time=",classify_time-tfidf_time
#
#'''-----------------------predict-------------------------'''
#testData_ = np.load('test_unigram_11_6.npy')
#testData = tfv.transform(listDown(testData_))
#pred = clf_ensemble.predict(testData)
#
#'''=========save the result for age/gender/education respectively====='''
submission = pd.read_csv('result_4.csv',sep=' ')
submission[label] = pred
submission.to_csv('result_4.csv',sep=' ',index=False)
#end_time=time.time()
##print "built csv time=",end_time-classify_time
#print "all time=",end_time-start_time
#
