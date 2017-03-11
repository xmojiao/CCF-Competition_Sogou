# -*- coding: utf-8 -*-
# @Time    : 2016/10/7 15:29
# @Author  : Aries
# @Site    : 
# @File    : TF.py
# @Software: PyCharm Community Edition
from __future__ import division

# tf 是该词在本篇中的概率,每一篇query分开处理
# Word_Repetition {'userid': {'gender':'cutWordList'},'userid':{'gender', 'cutWordList'}]
# 计算nomalTF
def tf(cutwords):
    TF = {}
    for userid , tem_cutwords in cutwords.iteritems():
        dict_TF = {}
        for label , listWord in tem_cutwords.iteritems():
            tem_dict_TF = {}
            dict_resTF = {}
            for word in listWord:
                if word in dict_resTF.keys():
                    dict_resTF[word] += 1
                else:
                    dict_resTF[word] = 1
            # 计算TF
            wordLen = len(listWord)
            for k, v in dict_resTF.iteritems():
                dict_TF[k] = v/wordLen
            tem_dict_TF[label] = dict_TF
        TF.setdefault(userid,tem_dict_TF)
    return TF

# 返回格式  userId  word TF  word TF
