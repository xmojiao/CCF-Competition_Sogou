# -*- coding: UTF-8 -*-
# @Time    : 2016/10/7 16:29
# @Author  : Aries
# @Site    : 
# @File    : IDF.py
# @Software: PyCharm Community Edition

import math

# 数据格式  userId word TF  word TF
# allTF {'userid': {'gender':{'ci':0.1}},'userid':{'gender',{ 'da':0.2}}]

def idf(allTF):
    userCount = len(allTF)
    dict_resIDF = {}
    for userid, v in allTF.iteritems():
        for label, v1 in v.iteritems():
            for word ,v2 in v1.iteritems():
                count = 0
                for _userid ,_v in allTF.iteritems():
                    for _label ,_v1 in _v.iteritems():
                        if word in _v1.keys():
                            count += 1
                dict_resIDF[word] = math.log10(userCount/count+0.01)
    return dict_resIDF

# do TF-IDF
# 如果特征词袋包含则计算。
# feature = {"柔和": 1, "0000格式": 2, "哈达": 3}

def tf_idf(allTF, dict_resIDF, feature):
    f = open(r'D:\CCFBDC\testResoult\test1.txt','w')
    for userid , v in allTF.iteritems():
        for label, tfDict in v.iteritems():
            f.write(label)
            for word , tf in tfDict.iteritems():
                if word in dict_resIDF.keys():
                    if word in feature.keys():
                        TF_IDF =  tf * dict_resIDF.get(word)
                        f.write( ' '+ str(feature.get(word)) + ':'+ str(round(TF_IDF,5)))
        f.write('\n')
    f.close()