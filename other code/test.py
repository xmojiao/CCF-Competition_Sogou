#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/10/26 11:06
# @Author  : Aries
# @Site    : 
# @File    : test.py
# @Software: PyCharm Community Edition
import csv
f  = open('test_word_bag/user_tag_query.2W.TEST')
f1 = open('resoult_age_11.txt')
f2 = open('resoult_gender_11.txt')
f3 = open('resoult_edu_11.txt')
csvfile = file('testResoult_svm_12.csv', 'wb')

userid = []
age = []
gender = []
edu = []

for line in f:
    # 编码转换and按空格切割、提取查询关键字、每个用户合并成为一句话
    single_line = line.decode('gbk', 'ignore').encode('utf-8').split()
    userid.append(single_line[:1])

for line in f1:
    # 编码转换and按空格切割、提取查询关键字、每个用户合并成为一句话
    single_line = line.decode('gbk', 'ignore').encode('utf-8').split()
    age.append(single_line[:1])

for line in f2:
    # 编码转换and按空格切割、提取查询关键字、每个用户合并成为一句话
    single_line = line.decode('gbk', 'ignore').encode('utf-8').split()
    gender.append(single_line[:1])

for line in f3:
    # 编码转换and按空格切割、提取查询关键字、每个用户合并成为一句话
    single_line = line.decode('gbk', 'ignore').encode('utf-8').split()
    edu.append(single_line[:1])


writer = csv.writer(csvfile)

for i in range(20000):
    writer.writerow([str(userid[i])[2:-2]+' '+str(age[i])[2:-2]+' '+str(gender[i])[2:-2]+' '+str(edu[i])[2:-2]])
csvfile.close()