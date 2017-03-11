# -*- coding: utf-8 -*-
"""
Created on Sun Oct 05 23:58:46 2016

@author: huanghe
"""
from sklearn import svm
#import jieba
#import jieba.analyse
#import kafang1
#import TF
#import IDF
import cPickle as pickle
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets.base import Bunch

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法
import numpy as np
from sklearn import metrics

from sklearn.feature_selection import SelectKBest, chi2
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
                if check not in stopwords and single_line[3]!='0':
                    user_word = user_word + ' '+check
        if single_line[3]!='0':
            new_item = [single_line[0],single_line[3],user_word]
            new_table.append(new_item)
    #

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

def calculate_result(actual,pred):
    c=0.000
    d=0.000
    for i in range(len(actual)):
        if actual[i]==pred[i]:
            c+=1
    d=float(c)/len(actual)
#    print 'right:{0:.3f}'.format(d) 
    return d

#data_train  = preprocess('user_tag_query.2W.TRAIN')
data_train = readbunchobj('seg_edu_train.txt')
#writebunchobj('seg_edu_train.txt', data_train)
#data_test  = preprocess1('user_tag_query.2W.TEST')
#writebunchobj('seg_edu_test.txt', data_test)
data_test = readbunchobj('seg_edu_train.txt')
y_train = data_train.label
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(data_train.contents)
print('--------------2')
X_test = vectorizer.transform(data_test.contents)
ch2 = SelectKBest(chi2, k=40000)
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)

y_train=y_train[:17000]
y_test=y_train[17000:]
X_train=X_train[:17000]
X_test=X_train[17000:]

#model = svm.LinearSVC(C=0.1,penalty="l1", dual=False, tol=1e-3)
model = svm.LinearSVC(C=0.1)

model.fit(X_train, y_train)
predicted = model.predict(X_test)
total = len(predicted)
#print total
calculate_result(y_test,predicted)
#f = open('resoult_edu_11.txt','wb')
#
#for flabel, file_name, expct_cate in zip(data_test.label, data_test.userid, predicted):
#    # print file_name, ": 实际类别:", flabel, " -->预测类别:", expct_cate
#    f.write(expct_cate)
#    f.write('\n')
#    if flabel != expct_cate:
#        rate += 1
#f.close()
#print "error rate:", float(rate) * 100 / float(total), "%"
#print "edu预测完毕!!!"


