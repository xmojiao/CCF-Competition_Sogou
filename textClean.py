# -*- coding: utf-8 -*-
"""
Created on Sun Oct 02 11:54:53 2016
@author: Lenovo
"""
import pandas as pd
import numpy as np
#import jieba.posseg as pseg
import jieba
import time

# load the stopkey  the 'gbk' here is to tranform into unicode 
f = open('stopkey.txt','r')   
lines = f.readlines()
stopkey=[line.strip().decode('gbk') for line in lines]
f.close()
add = [' ',]
stopkey = stopkey+add

# load the userdict (first lower all the words)
f = open('userdict.txt','r')
all = f.read()
f.close()
f = open('userdict.txt','w')
f.write(all.lower())
f.close()
jieba.load_userdict('userdict.txt')


# for unigram, input whole the str with the ',' for easy jieba cut    
def unigram(str_):
    after_cut = jieba.cut(str_)
    after_cut = list(after_cut)
    return [i for i in after_cut if i not in stopkey]
    
# for bigram, the query for a user is firstly transform into a list for easy bi_link    
def bi_link(list_):
    return [list_[i]+list_[i+1] for i in xrange(len(list_)-1)]

def bigram(list_):
    a =[]
    for i in list_: 
        seg_list = unigram(i)
        a = a + bi_link(seg_list) #
    return a

def search_gram(str_):
    seg_list = [i for i in jieba.cut_for_search(str_) if i not in stopkey]
    return seg_list


alphabeta = 'abcdefghijklmnopqrstuvwxyz'
def english_judge(str_):
    if len(str_)<=3:
        return 0
    for i in str_ :
        if i not in alphabeta:
            return 0
    return 1

def add_english(list_):
    add = ['english' for i in list_ if english_judge(i)]
    return list_+add

#============================= load the data===================
trainData = pd.read_csv('trainData.csv',sep=' ')
trainData = trainData['query'].values.tolist()
# substitute \t into ','(for easy jieba.cut); lower the word and strip the string
for i in np.arange(len(trainData)):
    trainData[i] = ','.join(trainData[i].split('\t')).lower().strip()
	
testData = pd.read_csv('testData.csv',sep=' ')
testData = testData['query'].values.tolist()
# substitute \t into ','(for easy jieba.cut); lower the word and strip the string
for i in np.arange(len(testData)):
    testData[i] = ','.join(testData[i].split('\t')).lower().strip()
#==============================unigram===========================

print 'process for trainData'
time1 = time.time()    
train_unigram = [unigram(i) for i in trainData]    

f = open('train_unigram.txt','w')
for i in train_unigram:
    f.write(' '.join(i).encode('utf-8')+'\n')
f.close()
'''
# add the english
train_unigram = [add_english(i) for i in train_unigram]
'''

'''
result = pd.DataFrame({'query':train_unigram})
result.to_csv('train_unigram_test.csv',sep=' ',index=False)
'''  
print time.time()-time1,'seconds used to construct train_unigram.'
print '----------------------------------------------------------' 
print 'process for testData'
del train_unigram,trainData
time1 = time.time()    
test_unigram = [unigram(i) for i in testData]  

  
'''
# add the english
train_unigram = [add_english(i) for i in test_unigram]
'''

'''
result = pd.DataFrame({'query':test_unigram})
result.to_csv('test_unigram_test.csv',sep=' ',index=False) 
'''
f = open('test_unigram.txt','w')
for i in test_unigram:
    f.write(' '.join(i).encode('utf-8')+'\n')
f.close()
print time.time()-time1,'seconds used to construct test_unigram.' 

'''
#==============================bigram===========================
# caution: can't operate directly on the result of unigram
time1 = time.time()    
for i in np.arange(len(trainData)):
    trainData[i] = trainData[i].split(',')
train_bigram = [bigram(i) for i in trainData]    
print time.time()-time1

np.save('train_bigram_11_12_1.npy',train_bigram)
#np.save('test_bigram_11_12.npy',train_bigram)
'''

'''
#==============================search_gram=======================
# caution: can't operate directly on the result of unigram

time1 = time.time()    
train_bigram = [search_gram(i) for i in trainData]    
print time.time()-time1

np.save('train_search_gram_11_12_1.npy',train_bigram)

#train_search_gram_cleaned = [stop_words_clean(i) for i in train_search_gram]
'''

        
        