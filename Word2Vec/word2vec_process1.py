# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 12:37:52 2016

@author: Jiao
"""

import numpy as np
import pandas as pd

'''将测试集按比例分配，用于测试'''
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale

def buildWordVector(text,size):
    vec=np.zeros(size).reshape((1,size))
    count=0.
    for word in text:
        try:
            vec+=imdb_w2v[word].reshape((1,size))
            count+=1.
        except KeyError:
            continue
    if count!=0:
        vec/=count
    return vec
def calculate_result(actual,pred):
    c=0.000
    d=0.000
    for i in range(len(actual)):
        if actual[i]==pred[i]:
            c+=1
    d=float(c)/len(actual)
#    print 'right:{0:.3f}'.format(d) 
    return d    
'''
函数功能：加载数据，并出去标签为0未知性别的部分
传入文件：样本切词后的文件和对应的label文件
输出文件：
'''
def data_load(label):
#    lines=open('test_unigram1.txt','r').readlines()
#    trainData_all=[i.strip() for i in lines]
#    filename='../sougodownload/rematch/trainData.txt'
#    lines=open(filename,'r').readlines()
#    trainData_all=[i.strip() for i in lines]

#    trainLabel_all = pd.read_csv('trainLabel.csv',sep=' ')
#    trainData_all = pd.read_csv('train_unigram.csv')
#    trainData_all = np.load('data_first/train_unigram_11_6.npy')
#    trainLabel_all = pd.read_csv('data_first/trainLabel.csv',sep=' ')
    trainData_all = np.load('train_unigram_11_6.npy')
    trainLabel_all = pd.read_csv('trainLabel.csv',sep=' ')

#    trainData_all = np.load('train_unigram_11_6.npy')
#    trainLabel_all = pd.read_csv('trainLabel.csv',sep=' ')
    trainLabel_all = np.array((trainLabel_all[label]))
    trainData_all=np.array(trainData_all)
    #remove the unknown data 
    known = [i for i in range(len(trainLabel_all)) if trainLabel_all[i]!=0]
    trainData_ = trainData_all[known]
    trainLabel = trainLabel_all[known]
    return trainData_,trainLabel
    
def data_Word2Vec(trainData):
    n_dim=200
    imdb_w2v=Word2Vec(size=n_dim,min_count=1)
    imdb_w2v.build_vocab(trainData)
    
    imdb_w2v.train(trainData)    
    return n_dim,imdb_w2v
    #train_vecs=scale(train_vecs)
        	
model=Word2Vec.load('Word60.model')  
#
##model=Word2Vec.load_word2vec_format('../word_files/w2v_2/trunk/vectors_delete.bin',binary=True) 
trainData,trainLabel=data_load('age') # 'age','gender','education'
n_dim,imdb_w2v= data_Word2Vec(trainData) 
train_vecs=np.concatenate([buildWordVector(z,n_dim) for z in trainData])
x_train,x_test,y_train,y_test=train_test_split(train_vecs,trainLabel,test_size=0.2)
#    
##from sklearn.naive_bayes import BernoulliNB, MultinomialNB
###''' NB, best parameters:  gender: 1e-8'''
##alpha_NB=2e-3
##clf_NB = BernoulliNB(alpha_NB)   #gender 2e-3  age 3  education 6e-5
##clf_NB.fit(x_train,y_train)                        %0.71
##pred = clf_NB.predict(x_test)
##z0=calculate_result(y_test,pred) 
##print 'alpha_NB=',alpha_NB,'  BernoulliNB=',z0
##
#''' MaxEnt, parameters untuned  gender:1.5---''' 
#from sklearn.linear_model import LogisticRegression
#clf_ME = LogisticRegression(n_jobs=-1) #gender 不变   %0.793
#clf_ME.fit(x_train,y_train)
#pred = clf_ME.predict(x_test)
#z2=calculate_result(y_test,pred)
#print ' MaxEnt=', z2#'n_jobs=',n_jobs,   vectors_part拟合%0.685

#


#from sklearn.ensemble import RandomForestClassifier
#clf=RandomForestClassifier(n_estimators=100,min_samples_split=2,random_state=0)
#clf.fit(x_train,y_train)                           #%0.778
#pred = clf.predict(x_test)
#z3=calculate_result(y_test,pred)
#print 'RandomForst',z3
#
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
print '*************************\nSVM\n**********************' 
C=0.1 
clf_SVM=SVC(kernel='linear',C=C)                       #%0.7939
#clf_SVM = CalibratedClassifierCV(svclf)
clf_SVM.fit(x_train,y_train)
pred =clf_SVM.predict(x_test)
z4=calculate_result(y_test,pred)
print 'C=',C,'  SVM=',z4
##
# predict
#clf_SVM.fit(train_vecs,trainLabel)
#testData = pd.read_csv('test_unigram.csv',sep=' ')
#testData=np.array(testData)
#n_dim,imdb_w2v= data_Word2Vec(testData) 
#test_vecs=np.concatenate([buildWordVector(z,n_dim) for z in testData])
##testData_ = np.load('test_unigram.npy')
#pred = clf_SVM.predict(test_vecs)   #预测标签
#
##=========save the result for age/gender/education respectively=====
#submission = pd.read_csv('testID.csv',sep=' ')
#submission['age'] = pred
#submission.to_csv('testID.csv',sep=' ',index=False)
#
#
#from NNet import NeuralNet
#clf_nnet=NeuralNet(50,learn_rate=1e-2)  #            %0.79
#maxiter=500
#batch=150
#clf_nnet.fit(x_train,y_train,fine_tune=False,maxiter=maxiter,SGD=True,batch=batch,rho=0.9)
#print 'Maxiter=',maxiter
#print 'ANN=: %.2f'%clf_nnet.score(x_test,y_test)
#pred =clf_nnet.predict(x_test)
#z5=calculate_result(y_test,pred)
#print 'Maxiter=',maxiter,'  ANN=',z5
#
#'''==========================ensemble==========================='''
#from sklearn.ensemble import VotingClassifier
#clf_ensemble = VotingClassifier(estimators=[('MNB', clf), ('ME', clf_ME),\
#                                     ('SVM',clf_SVM)], voting='soft')
#clf_ensemble.fit(x_train,y_train)
#pred = clf_ensemble.predict(x_test)
##print "Mean classify: ",(z0+z1+z2+z3)/4
#print "ensemble right: ",calculate_result(y_test,pred)

#print "Mean classify: ",(z0+z1+z2+z3)/4


#from sklearn.linear_model import SGDClassifier
#lr=SGDClassifier(loss='log',penalty='l1')
#lr.fit(train_vecs,y_train)
#print 'Test Accuracy: %.2f'%lr.score(test_vecs,y_test)

#from sklearn.metrics import roc_curve,auc
#import matplotlib.pyplot as plt
#pred_probas=lr.predict_proba(x_test)[:,1]
#fpr,tpr,_=roc_curve(y_test,pred_probas)
#roc_auc=auc(fpr,tpr)
#plt.plot(fpr,tpr,label='area=%.2f'%roc_auc)
#plt.plot([0,1],[0,1],'k--')
#plt.xlim([0.0,1.0])
#plt.ylim([0.0,1.0])
#plt.legend(loc='lower right')
#plt.show()
