# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 22:25:56 2016

@author: Jiao
"""

from sklearn import svm
import jieba
#import jieba.analyse
#import kafang1
#import TF
#import IDF
import numpy as np
import pandas as pd
import cPickle as pickle
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets.base import Bunch
#from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
#from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法
#import numpy as np
#from sklearn import metrics
#
#from sklearn.feature_selection import SelectKBest, chi2
#from sklearn.pipeline import Pipeline
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import BernoulliNB, MultinomialNB
#from sklearn.linear_model import RidgeClassifier
#from sklearn.linear_model import Perceptron
#from sklearn.neighbors import NearestCentroid
#from sklearn.linear_model import SGDClassifier
#from sklearn.svm import LinearSVC
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn import metrics
#from time import time
#import re
#import jieba
#import jieba.analyse
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
def readbunchobj(path):
    file_obj = open(path, "rb")
    bunch = pickle.load(file_obj)
    file_obj.close()
    return bunch


def writebunchobj(path, bunchobj):
    file_obj = open(path, "wb")
    pickle.dump(bunchobj, file_obj)
    file_obj.close()

def preprocess(path):
    #读入停词表、构建停词字典
    stoptxt = open('stopwords.txt')
    word = []
    for line in stoptxt:
        word.append(line.decode('gb18030','ignore').encode('utf-8').strip('\n'))
    stopwords = {}.fromkeys([line.rstrip() for line in word])

    new_table = []
    f = open(path,'r')
    for line in f:
        #编码转换and按空格切割、提取查询关键字、每个用户合并成为一句话
        single_line = line.decode('gbk','ignore').encode('utf-8').split()
        key_words = single_line[4:]
        user_word = ''
        for item in key_words:
            seg_list = jieba.cut(item)
            for check in seg_list:
                check = check.encode('utf-8')
                if check not in stopwords and single_line[2]!='0':
                    user_word = user_word + ' '+check
        if single_line[2]!='0':
            new_item = [single_line[0],single_line[2],user_word]
            new_table.append(new_item)

    bunch = Bunch(label=[], userid=[], contents=[])
    for k in new_table:
        bunch.userid.append(k[0])
        bunch.label.append(k[1])
        bunch.contents.append(k[2])
    return bunch
def preprocess1(path):
    # 读入停词表、构建停词字典
    stoptxt = open('stopwords.txt')
    word = []
    for line in stoptxt:
        word.append(line.decode('gb18030', 'ignore').encode('utf-8').strip('\n'))
    stopwords = {}.fromkeys([line.rstrip() for line in word])
    new_table = []
    f = open(path, 'r')
    for line in f:
        # 编码转换and按空格切割、提取查询关键字、每个用户合并成为一句话
        single_line = line.decode('gbk', 'ignore').encode('utf-8').split()
        key_words = single_line[1:]
        user_word = ''
        for item in key_words:
            seg_list = jieba.cut(item)
            for check in seg_list:
                check = check.encode('utf-8')
                if check not in stopwords:
                    user_word = user_word + ' ' + check
        new_item = [single_line[0], user_word]
        new_table.append(new_item)
    bunch = Bunch(label=[], userid=[], contents=[])
    for k in new_table:
        bunch.userid.append(k[0])
        bunch.label.append('1')# 全部设为1
        bunch.contents.append(k[1])
    return bunch

#data_train  = preprocess('user_tag_query.2W.TRAIN')
#a=np.array([1,2,5,5,6])
#data_test  = preprocess1('user_tag_query.2W.TEST')
#writebunchobj('seg_edu_test.txt', data_test)
def calculate_result(actual,pred):
    c=0.000
    d=0.000
    for i in range(len(actual)):
        if actual[i]==pred[i]:
            c+=1
    d=float(c)/len(actual)
#    print 'right:{0:.3f}'.format(d) 
    return d

#data_train  = preprocess('user_train_part18000.csv')
#writebunchobj('part_gender_train.txt', data_train)
#data_test  = preprocess('user_train_part2000.csv')
#writebunchobj('part_gender_test.txt', data_test)

data_train = readbunchobj('part_gender_train.txt')
data_test = readbunchobj('part_gender_test.txt')
trainLabel, trainLabel_test = data_train.label, data_test.label
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train= vectorizer.fit_transform(data_train.contents)
X_test = vectorizer.transform(data_test.contents)
ch2 = SelectKBest(chi2, k=40000)
trainData = ch2.fit_transform(X_train, trainLabel)
trainData_test_ = ch2.transform(X_test)
 


'''-- SVM, transform into probability output--'''
from sklearn.calibration import CalibratedClassifierCV
C=0.15
clf_= svm.LinearSVC(C=0.15) #gender c=0.15   age 0.1
clf_SVM = CalibratedClassifierCV(clf_)
clf_SVM.fit(trainData,trainLabel)
pred = clf_SVM.predict(trainData_test_)
z3=calculate_result(trainLabel_test,pred) 
#print 'C=',C,'  SVM=',z3
'''-----------------------predict-------------------------'''
#test_data  = preprocess1('user_tag_query.2W.TEST') 
#writebunchobj('seg_gender_test.txt', test_data)

test_data = readbunchobj('seg_gender_test.txt')
testData = vectorizer.transform(test_data.contents)
testData = ch2.transform(testData)
#
#pred = clf_SVM.predict(testData)
#
#'''=========save the result for age/gender/education respectively====='''
submission = pd.read_csv('result_devised1.csv',sep=' ')
submission['gender'] = pred
submission.to_csv('result_devised1.csv',sep=' ',index=False)
#
##''' NB, best parameters:  gender: 1e-8'''
#alpha_NB=0.08
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
''' MaxEnt, parameters untuned  gender:1.5---''' 
#from sklearn.linear_model import LogisticRegression
#clf_ME = LogisticRegression(n_jobs=-5) #gender 不变
#clf_ME.fit(trainData,trainLabel)
#pred = clf_ME.predict(trainData_test_)
#z2=calculate_result(trainLabel_test,pred)
#print ' MaxEnt=', z2#'n_jobs=',n_jobs,
#

#from sklearn.ensemble import AdaBoostClassifier
#clf_=AdaBoostClassifier(n_estimators=1000)
#clf_Ada = CalibratedClassifierCV(clf_)
#clf_Ada.fit(trainData,trainLabel)
#pred = clf_Ada.predict(trainData_test_)
#z3=calculate_result(trainLabel_test,pred) 
#print 'n_estimators=100','  Adaboost=',z3
#from sklearn.ensemble import RandomForestClassifier
#clf=RandomForestClassifier(n_estimators=10,random_state=0)
#clf.fit(trainData,trainLabel)
#pred = clf.predict(trainData_test_)
#z3=calculate_result(trainLabel_test,pred)
#print 'RandomForst=',z3


'''单个算法调参后，使用votingClassifier投票'''
##'''==========================ensemble==========================='''
#from sklearn.ensemble import VotingClassifier
#clf_ensemble = VotingClassifier(estimators=[('NB', clf_NB), ('MNB', clf_MNB), ('ME', clf_ME),\
#                                     ('SVM',clf_SVM)], voting='soft')
#clf_ensemble.fit(trainData,trainLabel)
#pred = clf_ensemble.predict(trainData_test_)
##print "Mean classify: ",(z0+z1+z2+z3)/4
#print "ensemble right: ",calculate_result(trainLabel_test,pred)
#
##classify_time=time.time()
##print "classify time=",classify_time-tfidf_time
#


