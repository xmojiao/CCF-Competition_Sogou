# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 17:14:51 2016

@author: Jiao
"""
from gensim.models import Word2Vec


def getWordVecs(words):
    vecs=[]
    for word in words:
#        word=word.replace('\n','')
        try:
            vecs.append(model[word].reshape((1,60)))
        except KeyError:
            continue
    vecs=np.concatenate(vecs)
    return np.array(vecs,dtype='float')

food_word=[u'猪肉',u'西红柿',u'土豆',u'面条']
weather_word=[u'阴天',u'雪',u'晴天',u'雨']
sport_word=[u'篮球',u'足球',u'排球',u'羽毛球']
#留一法交叉验证，test_size为测试集所占的比例
model=Word2Vec.load('Word60.model')
model=Word2Vec.load('Word60.model')
food_vecs=getWordVecs(food_word)
weather_vecs=getWordVecs(weather_word)
sport_vecs=getWordVecs(sport_word)
from sklearn.cross_validation import train_test_split
#from gensim.models.word2vec import Word2Vec
y= np.concatenate((np.ones(len(food_vecs)),np.zeros(len(weather_vecs))))

x_train,x_test,y_train,y_test=train_test_split(np.concatenate((food_vecs,weather_vecs)),y,test_size=0.2)
from sklearn import metrics
def calculate_result(actual,pred):
    m_precision=metrics.precision_score(actual,pred)
#    m_recall=metrics.recall_score(actual,pred)
    print 'precision:{0:.3f}'.format(m_precision)
from sklearn.svm import SVC
print '*************************\nSVM\n*************************'  
svclf=SVC(kernel='linear',C=0.2)
svclf.fit(x_train,y_train)
pred =svclf.predict(x_test)
calculate_result(y_test,pred)
''' 降维之后样本的分布'''
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
ts=TSNE(2)
reduced_vecs=ts.fit_transform(np.concatenate((food_vecs,weather_vecs,sport_vecs)))
for i in range(len(reduced_vecs)):
    if i <len(food_vecs):
        color='b'
    elif i>=len(food_vecs) and i<len(food_vecs)+len(weather_vecs):
        color='r'
    else:
        color='g'
    plt.plot(reduced_vecs[i,0],reduced_vecs[i,1],marker='o',color=color,markersize=8)
y=np.concatenate((np.ones(len(food_vecs)),np.zeros(len(weather_vecs))))  

#''' MNB, best parameters: gender: 1e-8 age:1e-3 education:1e-2 '''
#alpha_MNB=4e-2
#clf_MNB = MultinomialNB(alpha_MNB)    #gender  4e-2 age 0.09
#clf_MNB.fit(x_train,y_train)
#pred = clf_MNB.predict(x_test)
#z1=calculate_result(y_test,pred) 
#print 'alpha_MNB=',alpha_MNB,'  MultinomialNB=',z1
#
#'''---------------------KMeans--------------------'''
#from sklearn.cluster import KMeans  
#print '*************************\nKMeans\n*************************'  
#pred = KMeans(n_clusters=5)  
#pred.fit(x_test)  
#z5=calculate_result(y_test,pred.labels_);  
#print '  KMeans=',z5
#
#'''--------------KNN-------'''
#from sklearn.naive_bayes import  KNeighborsClassifier
#print '*************************\nKNN\n*************************' 
#knnclf = KNeighborsClassifier()#default with k=5  
#knnclf.fit(x_train,y_train)
#pred = knnclf.predict(x_test);  
#z6=calculate_result(y_test,pred); 
#print '  KNN=',z6
