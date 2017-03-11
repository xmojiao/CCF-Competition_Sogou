# -*- coding: utf-8 -*-
"""
Created on Sun Oct 05 23:58:46 2016

@author: huanghe
"""
import numpy as np
import pandas as pd
import time
import pickle
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import svm
#from sklearn.model_selection import GridSearchCV   # too time consuming
from sklearn.cross_validation import cross_val_score#cross_validation
from sklearn.cross_validation import StratifiedKFold
#from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import hamming_loss

def listDown(s):
    t = []
    for i in np.arange(len(s)):
        t.append(' '.join(s[i]))
    return t

def bi_change(a):           
    b = []
    for i in np.arange(len(a)):
	    x,y = a[i]
	    b.append(x+y)
    return b
    
def tri_change(a):           
    b = []
    for i in np.arange(len(a)):
	    x,y,z = a[i]
	    b.append(x+y+z)
    return b

feature_number = 20000
bigram_feature_number = 10
trigram_feature_number = 0
#load the data , change the label into 'age','gender','education'
label = 'age'	
trainData_all = np.load('trainData_afterClean_3.npy')
trainLabel_all = pd.read_csv('trainLabel.csv',sep=' ')
trainLabel_all = np.array((trainLabel_all[label]))

#remove the unknown data 
known = [i for i in range(len(trainLabel_all)) if trainLabel_all[i]!=0]
trainData_ = trainData_all[known]
trainLabel = trainLabel_all[known]
'''-----------------截取三个样本----------'''
trainData_ = trainData_[:10]

trainLabel = trainLabel[:10]
'''
trainData_= trainData_[0:200]
trainLabel = trainLabel[0:200]
'''

###### NLTK feature selection########### 
from nltk.metrics import BigramAssocMeasures,TrigramAssocMeasures
from nltk.collocations import BigramCollocationFinder,TrigramCollocationFinder
                              
'''  #use all words as feature, unnecessay.  implemented by handy comment
def word_feats(words):  
    return dict([(word, True) for word in words])   
print 'evaluating single word features' 
'''  

def create_word_scores():
    time1 = time.time()

    word_fd = {} #可统计所有词的词频
    cond_word_fd = {}  #可统计积极文本中的词频和消极文本中的词频
    
    total_word_count = 0
    for i in list(np.unique(trainLabel)):
        labelIndex = [j for j in np.arange(len(trainLabel)) if trainLabel[j]==i]
        labelWords = trainData_[labelIndex[0]]
        for j in np.arange(1,len(labelIndex)):
            labelWords = labelWords + trainData_[labelIndex[j]]
                   
        cond_word_fd[str(i)]={}
        for word in labelWords:
            if word not in word_fd:
                word_fd[word]=1
            else:
                word_fd[word] += 1
            if word not in cond_word_fd[str(i)]:
                cond_word_fd[str(i)][word]=1
            else:
                cond_word_fd[str(i)][word] += 1
                
        label_word_count = len(cond_word_fd[str(i)]) #积极词的数量
        total_word_count = total_word_count + label_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        word_scores[word]=0   
        for k in list(np.unique(trainLabel)):
            if word in cond_word_fd[str(k)]:
                word_scores[word] = word_scores[word] + BigramAssocMeasures.chi_sq(cond_word_fd[str(k)][word],\
                                   (freq, len(cond_word_fd[str(k)])), total_word_count)
            else:
                word_scores[word] = word_scores[word] + BigramAssocMeasures.chi_sq(0,\
                                   (freq, len(cond_word_fd[str(k)])), total_word_count)
                
    time2 = time.time()
    print time2 - time1                           
    return word_scores #包括了每个词和这个词的信息量

def find_best_words(word_scores, number):    
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, \
                reverse=True)[:number] #把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的    
    best_words = set([w for w, s in best_vals])    
    return best_words
    
def best_word_feats(words):  
    return dict([(word, True) for word in words if word in find_best_words])  

#may be useful for Chinese 
'''-------使用了双次搭配作为特征-------'''   
def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=bigram_feature_number):  
    bigram_finder = BigramCollocationFinder.from_words(words)  
    bigrams = bigram_finder.nbest(score_fn, n)  
    d = dict([(bigram, True) for bigram in bigrams])  
    #d.update(best_word_feats(words))  
    return d 

def best_trigram_word_feats(words, score_fn=TrigramAssocMeasures.chi_sq, n=trigram_feature_number):
    trigram_finder = TrigramCollocationFinder.from_words(words)  
    trigrams = trigram_finder.nbest(score_fn, n)  
    d = dict([(trigram, True) for trigram in trigrams])  
    #d.update(best_word_feats(words))  
    return d 

'''
def best_quadgram_word_feats(words, score_fn=QuadgramAssocMeasures.chi_sq, n=trigram_feature_number):
    trigram_finder = TrigramCollocationFinder.from_words(words)  
    trigrams = trigram_finder.nbest(score_fn, n)  
    d = dict([(trigram, True) for trigram in trigrams])  
    #d.update(best_word_feats(words))  
    return d 
'''

word_scores = create_word_scores()
si_keys = list(find_best_words(word_scores,feature_number))  # a list of most informative words
print "Chi2 Done!"
'''----------------------------'''
keys = list(set(si_keys)) #将set传换成list
values = range(len(keys)) #从0到len（keys）
dict_fromChi2 = dict(zip(keys, values))     #将keys,values打包成一个字典


'''
print "save the Chi2 word_scores for "+label
f=file('word_scores_'+label+'_'+str(feature_number)+'.txt','wb')
pickle.dump(word_scores,f)
f.close()
'''


## preprocess the trainData_ to feed in the function
feedData = trainData_[0]
for i in np.arange(1,len(trainData_)):
    feedData = feedData + trainData_[i]
bi_best_feature = best_bigram_word_feats(feedData)  #this input may be augumented by including test
tri_best_feature = best_trigram_word_feats(feedData)
bi_keys = bi_best_feature.keys()
bi_keys = list(set(bi_change(bi_keys)))
tri_keys = tri_best_feature.keys()
tri_keys = list(set(tri_change(tri_keys)))
print len(bi_keys),'bigram and',len(tri_keys),'trigram'


keys = list(set(si_keys+bi_keys+tri_keys))
values = range(len(keys))
dict_fromChi2 = dict(zip(keys, values))


feedData = []
for i in np.arange(len(trainData_)):
    feedData.append(trainData_[i]+bi_change(best_bigram_word_feats(trainData_[i]).keys())\
                    + tri_change(best_trigram_word_feats(trainData_[i]).keys()))

	

###########TFIDF feature extraction###########
tfv = TfidfVectorizer(min_df=3,max_df=0.5,vocabulary = dict_fromChi2)
trainData = tfv.fit_transform(listDown(feedData))


'''
################ apply PCA
print "applying PCA..."
trainData = trainData.toarray()
trainData = trainData.toarray - np.mean(trainData,axis = 0)      # zero mean
pca = PCA(n_components=0.90, svd_solver='full')
trainData = pca.fit_transform(trainData)
print "the variance of each component:"
print(pca.explained_variance_ratio_)
'''

'''
###########Hashvectorizer###############
vectorizer = HashingVectorizer(stop_words = 'english',non_negative = True, n_features = 10000)  
trainData = vectorizer.fit_transform(listDown(feedData))
'''

'''
### one v.s one
clf = OneVsOneClassifier(svm.LinearSVC(random_state=0,C=0.2))

'''

########classification  best(linerSVC) gender:c=0.2  age:c=0.25
#  linearSVC take a OVR tragety for multi-classification
#clf = svm.SVC(kernel = 'linear', C = 1)
#clf = BernoulliNB()
#clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
#clf = RandomForestClassifier(n_estimators=10)

#clf.fit(trainData,trainLabel)
print 'one vs. rest'
clf = svm.LinearSVC(C=0.2)
temp = cross_val_score(clf,trainData,trainLabel, cv=5)
print temp
print temp.mean()

print 'one vs. one'
clf_ = OneVsOneClassifier(svm.LinearSVC(random_state=0,C=0.2))
temp = cross_val_score(clf_,trainData,trainLabel, cv=5)
print temp
print temp.mean()
print '\n'

print 'balanced class weight'
print 'one vs. rest'
clf = svm.LinearSVC(C=0.2,class_weight = 'balanced')
temp = cross_val_score(clf,trainData,trainLabel, cv=5)
print temp
print temp.mean()

print 'one vs. one'
clf_ = OneVsOneClassifier(svm.LinearSVC(random_state=0,C=0.2,class_weight = 'balanced'))
temp = cross_val_score(clf_,trainData,trainLabel, cv=5)
print temp
print temp.mean()
#print(cross_val_score(clf,trainData,trainLabel, cv=3, fit_params = \
#      {sample_weight: sample_weight}))

print "above approach is the best so far... "
print "\n add the create featrue"

feature_max = np.max(trainData.toarray(),axis = 1)
feature_sum = np.sum(trainData.toarray(),axis = 1)
feature_all = np.hstack((trainData,feature_max[:,np.newaxis],feature_sum[:,np.newaxis]))

print 'one vs. rest'
clf = svm.LinearSVC(C=0.2)
temp = cross_val_score(clf,feature_all,trainLabel, cv=5)
print temp
print temp.mean()

print 'one vs. one'
clf_ = OneVsOneClassifier(svm.LinearSVC(random_state=0,C=0.2))
temp = cross_val_score(clf_,feature_all,trainLabel, cv=5)
print temp
print temp.mean()
print '\n'

print 'balanced class weight'
print 'one vs. rest'
clf = svm.LinearSVC(C=0.2,class_weight = 'balanced')
temp = cross_val_score(clf,feature_all,trainLabel, cv=5)
print temp
print temp.mean()

print 'one vs. one'
clf_ = OneVsOneClassifier(svm.LinearSVC(random_state=0,C=0.2,class_weight = 'balanced'))
temp = cross_val_score(clf_,feature_all,trainLabel, cv=5)
print temp
print temp.mean()
# cross validation is too time conusming
'''
tuned_parameters = [\
                      {'C': [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45]}]

clf_ = GridSearchCV(estimator=clf, param_grid=tuned_parameters, cv = 5,\
                               n_jobs = -1)
clf_.fit(trainData,trainLabel)

print clf_.best_score_
print clf_.best_estimator.C 
'''
'''
### prediction

testData_ = np.load('testData_afterClean.npy')
testData = tfv.transform(listDown(testData_))

testLabel = clf_.predict(testData)

'''
