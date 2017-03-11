# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 17:11:55 2016

@author: Lenovo
"""
import pandas as pd
import numpy as np
import time 
import jieba

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
def unigram(str_):
    after_cut = jieba.cut(str_)
    after_cut = list(after_cut)
    return [i for i in after_cut if i not in stopkey]
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

time1 = time.time()
f = open('../sougodownload/rematch/user_tag_query.10W.TRAIN')
lines = f.readlines()

data = []
label = []
for line in lines:
    temp = line.split('\t',4)
    data.append(temp[4])
    label.append(temp[:4])
f.close()

'''------------------------------------------'''
trainLabel = pd.DataFrame(np.array(label),columns=['id','age','gender','education'])
trainLabel.to_csv('../sougodownload/rematch/trainLabel.csv',sep=' ',index=False)

for i in np.arange(len(data)):
    data[i] = ','.join(data[i].split('\t')).lower().strip()

train_unigram = [unigram(i) for i in data]    
#''' add the english'''
#train_unigram = [add_english(i) for i in train_unigram]

'''写进txt格式文件中'''
f = open('../sougodownload/rematch/trainData.txt','w')
for i in train_unigram:
    f.write(' '.join(i).encode('utf-8')+'\n')
f.close()
'''写进csv格式中，并且加入行号维quary
trainData = pd.DataFrame({'query':data})
trainData.to_csv('../sougodownload/rematch/trainData.csv',sep=' ',index=False)
'''
print time.time()-time1,'seconds used to split the trainLabel and trainData' 

print '---------------------------------------------------------------'

time2 = time.time()
f = open('../sougodownload/rematch/user_tag_query.10W.TEST')
lines = f.readlines()

id = []
test = []
for line in lines:
    temp=line.split('\t',1)
    id.append(temp[0])
    test.append(temp[1])
f.close()

testID = pd.DataFrame({'id':id})
testID.to_csv('../sougodownload/rematch/testID.csv',sep=' ',index=False)


for i in np.arange(len(test)):
    test[i] = ','.join(test[i].split('\t')).lower().strip()

train_unigram = [unigram(i) for i in test]    
#''' add the english'''
#train_unigram = [add_english(i) for i in train_unigram]

'''写进txt格式文件中'''
f = open('../sougodownload/rematch/testData.txt','w')
for i in train_unigram:
    f.write(' '.join(i).encode('utf-8')+'\n')
f.close()

#testData = pd.DataFrame({'query':test})
#testData.to_csv('../sougodownload/rematch/testData.csv',sep=' ',index=False)
print time.time()-time2,'seconds used to split the testID and testData' 

